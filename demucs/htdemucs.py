# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# First author is Simon Rouard.
"""
This code contains the spectrogram and Hybrid version of Demucs.
"""
import math

from openunmix.filtering import wiener
import torch
from torch import nn
from torch.nn import functional as F
from fractions import Fraction
from einops import rearrange

from .transformer import CrossTransformerEncoder

from .demucs import rescale_module
from .states import capture_init
from .spec import spectro, ispectro
from .hdemucs import pad1d, ScaledEmbedding, HEncLayer, MultiWrap, HDecLayer


class HTDemucs(nn.Module):
    """
    Spectrogram and hybrid Demucs model.
    The spectrogram model has the same structure as Demucs, except the first few layers are over the
    frequency axis, until there is only 1 frequency, and then it moves to time convolutions.
    Frequency layers can still access information across time steps thanks to the DConv residual.

    Hybrid model have a parallel time branch. At some layer, the time branch has the same stride
    as the frequency branch and then the two are combined. The opposite happens in the decoder.

    Models can either use naive iSTFT from masking, Wiener filtering ([Ulhih et al. 2017]),
    or complex as channels (CaC) [Choi et al. 2020]. Wiener filtering is based on
    Open Unmix implementation [Stoter et al. 2019].

    The loss is always on the temporal domain, by backpropagating through the above
    output methods and iSTFT. This allows to define hybrid models nicely. However, this breaks
    a bit Wiener filtering, as doing more iteration at test time will change the spectrogram
    contribution, without changing the one from the waveform, which will lead to worse performance.
    I tried using the residual option in OpenUnmix Wiener implementation, but it didn't improve.
    CaC on the other hand provides similar performance for hybrid, and works naturally with
    hybrid models.

    This model also uses frequency embeddings are used to improve efficiency on convolutions
    over the freq. axis, following [Isik et al. 2020] (https://arxiv.org/pdf/2008.04470.pdf).

    Unlike classic Demucs, there is no resampling here, and normalization is always applied.
        混合时频域Demucs模型 (Hybrid Time-Frequency Demucs)
    
    这是一个结合了频谱图和时域波形处理的混合音源分离模型。模型的核心思想是：
    1. 频域分支：处理STFT频谱图，擅长捕捉频率特征和谐波结构
    2. 时域分支：直接处理波形，擅长捕捉瞬态信号和时间细节
    3. 跨域融合：通过Transformer实现两个分支的信息交互
    
    架构特点：
    - 双分支并行处理：频域编码器处理频谱，时域编码器处理波形
    - 跨域注意力：CrossTransformer在特征层面融合时频信息
    - 多尺度处理：通过多层编解码器实现多分辨率特征提取
    - 频率嵌入：为频域分支添加位置编码，提升频率轴卷积效率
    - 灵活输出：支持复数通道(CaC)、维纳滤波等多种后处理方式
    
    模型流程：
    输入混合音频 → 双分支预处理 → 并行编码 → Transformer融合 → 并行解码 → 后处理融合 → 分离音源
    
    与传统Demucs的差异：
    - 无重采样：直接在原始采样率下工作
    - 强制标准化：总是进行标准化处理
    - 混合架构：结合时域和频域的优势
    """

    @capture_init
    def __init__(
        self,
        sources,
        # Channels
        audio_channels=1,
        channels=48,
        channels_time=None,
        growth=2,
        # STFT
        nfft=4096,
        wiener_iters=0,
        end_iters=0,
        wiener_residual=False,
        cac=True,
        # Main structure
        depth=4,
        rewrite=True,
        # Frequency branch
        multi_freqs=None,
        multi_freqs_depth=3,
        freq_emb=0.2,
        emb_scale=10,
        emb_smooth=True,
        # Convolutions
        kernel_size=8,
        time_stride=2,
        stride=4,
        context=1,
        context_enc=0,
        # Normalization
        norm_starts=4,
        norm_groups=4,
        # DConv residual branch
        dconv_mode=1,
        dconv_depth=2,
        dconv_comp=8,
        dconv_init=1e-3,
        # Before the Transformer
        bottom_channels=0,
        # Transformer
        t_layers=5,
        t_emb="sin",
        t_hidden_scale=4.0,
        t_heads=8,
        t_dropout=0.0,
        t_max_positions=10000,
        t_norm_in=True,
        t_norm_in_group=False,
        t_group_norm=False,
        t_norm_first=True,
        t_norm_out=True,
        t_max_period=10000.0,
        t_weight_decay=0.0,
        t_lr=None,
        t_layer_scale=True,
        t_gelu=True,
        t_weight_pos_embed=1.0,
        t_sin_random_shift=0,
        t_cape_mean_normalize=True,
        t_cape_augment=True,
        t_cape_glob_loc_scale=[5000.0, 1.0, 1.4],
        t_sparse_self_attn=False,
        t_sparse_cross_attn=False,
        t_mask_type="diag",
        t_mask_random_seed=42,
        t_sparse_attn_window=500,
        t_global_window=100,
        t_sparsity=0.95,
        t_auto_sparsity=False,
        # ------ Particuliar parameters
        t_cross_first=False,
        # Weight init
        rescale=0.1,
        # Metadata
        samplerate=44100,
        segment=10,
        use_train_segment=True,
    ):
        """
        Args:
            sources (list[str]): list of source names.
            audio_channels (int): input/output audio channels.
            channels (int): initial number of hidden channels.
            channels_time: if not None, use a different `channels` value for the time branch.
            growth: increase the number of hidden channels by this factor at each layer.
            nfft: number of fft bins. Note that changing this require careful computation of
                various shape parameters and will not work out of the box for hybrid models.
            wiener_iters: when using Wiener filtering, number of iterations at test time.
            end_iters: same but at train time. For a hybrid model, must be equal to `wiener_iters`.
            wiener_residual: add residual source before wiener filtering.
            cac: uses complex as channels, i.e. complex numbers are 2 channels each
                in input and output. no further processing is done before ISTFT.
            depth (int): number of layers in the encoder and in the decoder.
            rewrite (bool): add 1x1 convolution to each layer.
            multi_freqs: list of frequency ratios for splitting frequency bands with `MultiWrap`.
            multi_freqs_depth: how many layers to wrap with `MultiWrap`. Only the outermost
                layers will be wrapped.
            freq_emb: add frequency embedding after the first frequency layer if > 0,
                the actual value controls the weight of the embedding.
            emb_scale: equivalent to scaling the embedding learning rate
            emb_smooth: initialize the embedding with a smooth one (with respect to frequencies).
            kernel_size: kernel_size for encoder and decoder layers.
            stride: stride for encoder and decoder layers.
            time_stride: stride for the final time layer, after the merge.
            context: context for 1x1 conv in the decoder.
            context_enc: context for 1x1 conv in the encoder.
            norm_starts: layer at which group norm starts being used.
                decoder layers are numbered in reverse order.
            norm_groups: number of groups for group norm.
            dconv_mode: if 1: dconv in encoder only, 2: decoder only, 3: both.
            dconv_depth: depth of residual DConv branch.
            dconv_comp: compression of DConv branch.
            dconv_attn: adds attention layers in DConv branch starting at this layer.
            dconv_lstm: adds a LSTM layer in DConv branch starting at this layer.
            dconv_init: initial scale for the DConv branch LayerScale.
            bottom_channels: if >0 it adds a linear layer (1x1 Conv) before and after the
                transformer in order to change the number of channels
            t_layers: number of layers in each branch (waveform and spec) of the transformer
            t_emb: "sin", "cape" or "scaled"
            t_hidden_scale: the hidden scale of the Feedforward parts of the transformer
                for instance if C = 384 (the number of channels in the transformer) and
                t_hidden_scale = 4.0 then the intermediate layer of the FFN has dimension
                384 * 4 = 1536
            t_heads: number of heads for the transformer
            t_dropout: dropout in the transformer
            t_max_positions: max_positions for the "scaled" positional embedding, only
                useful if t_emb="scaled"
            t_norm_in: (bool) norm before addinf positional embedding and getting into the
                transformer layers
            t_norm_in_group: (bool) if True while t_norm_in=True, the norm is on all the
                timesteps (GroupNorm with group=1)
            t_group_norm: (bool) if True, the norms of the Encoder Layers are on all the
                timesteps (GroupNorm with group=1)
            t_norm_first: (bool) if True the norm is before the attention and before the FFN
            t_norm_out: (bool) if True, there is a GroupNorm (group=1) at the end of each layer
            t_max_period: (float) denominator in the sinusoidal embedding expression
            t_weight_decay: (float) weight decay for the transformer
            t_lr: (float) specific learning rate for the transformer
            t_layer_scale: (bool) Layer Scale for the transformer
            t_gelu: (bool) activations of the transformer are GeLU if True, ReLU else
            t_weight_pos_embed: (float) weighting of the positional embedding
            t_cape_mean_normalize: (bool) if t_emb="cape", normalisation of positional embeddings
                see: https://arxiv.org/abs/2106.03143
            t_cape_augment: (bool) if t_emb="cape", must be True during training and False
                during the inference, see: https://arxiv.org/abs/2106.03143
            t_cape_glob_loc_scale: (list of 3 floats) if t_emb="cape", CAPE parameters
                see: https://arxiv.org/abs/2106.03143
            t_sparse_self_attn: (bool) if True, the self attentions are sparse
            t_sparse_cross_attn: (bool) if True, the cross-attentions are sparse (don't use it
                unless you designed really specific masks)
            t_mask_type: (str) can be "diag", "jmask", "random", "global" or any combination
                with '_' between: i.e. "diag_jmask_random" (note that this is permutation
                invariant i.e. "diag_jmask_random" is equivalent to "jmask_random_diag")
            t_mask_random_seed: (int) if "random" is in t_mask_type, controls the seed
                that generated the random part of the mask
            t_sparse_attn_window: (int) if "diag" is in t_mask_type, for a query (i), and
                a key (j), the mask is True id |i-j|<=t_sparse_attn_window
            t_global_window: (int) if "global" is in t_mask_type, mask[:t_global_window, :]
                and mask[:, :t_global_window] will be True
            t_sparsity: (float) if "random" is in t_mask_type, t_sparsity is the sparsity
                level of the random part of the mask.
            t_cross_first: (bool) if True cross attention is the first layer of the
                transformer (False seems to be better)
            rescale: weight rescaling trick
            use_train_segment: (bool) if True, the actual size that is used during the
                training is used during inference.
        """
        super().__init__()
        self.cac = cac
        self.wiener_residual = wiener_residual
        self.audio_channels = audio_channels
        self.sources = sources
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.bottom_channels = bottom_channels
        self.channels = channels
        self.samplerate = samplerate
        self.segment = segment
        self.use_train_segment = use_train_segment
        self.nfft = nfft
        self.hop_length = nfft // 4
        self.wiener_iters = wiener_iters
        self.end_iters = end_iters
        self.freq_emb = None
        assert wiener_iters == end_iters

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.tencoder = nn.ModuleList()
        self.tdecoder = nn.ModuleList()

        chin = audio_channels
        chin_z = chin  # number of channels for the freq branch
        if self.cac:
            chin_z *= 2
        chout = channels_time or channels
        chout_z = channels
        freqs = nfft // 2

        for index in range(depth):
            norm = index >= norm_starts
            freq = freqs > 1
            stri = stride
            ker = kernel_size
            if not freq:
                assert freqs == 1
                ker = time_stride * 2
                stri = time_stride

            pad = True
            last_freq = False
            if freq and freqs <= kernel_size:
                ker = freqs
                pad = False
                last_freq = True

            kw = {
                "kernel_size": ker,
                "stride": stri,
                "freq": freq,
                "pad": pad,
                "norm": norm,
                "rewrite": rewrite,
                "norm_groups": norm_groups,
                "dconv_kw": {
                    "depth": dconv_depth,
                    "compress": dconv_comp,
                    "init": dconv_init,
                    "gelu": True,
                },
            }
            kwt = dict(kw)
            kwt["freq"] = 0
            kwt["kernel_size"] = kernel_size
            kwt["stride"] = stride
            kwt["pad"] = True
            kw_dec = dict(kw)
            multi = False
            if multi_freqs and index < multi_freqs_depth:
                multi = True
                kw_dec["context_freq"] = False

            if last_freq:
                chout_z = max(chout, chout_z)
                chout = chout_z

            enc = HEncLayer(
                chin_z, chout_z, dconv=dconv_mode & 1, context=context_enc, **kw
            )
            if freq:
                tenc = HEncLayer(
                            chin,                    # 输入通道数
                            chout,                   # 输出通道数  
                            dconv=dconv_mode & 1,    # DConv残差分支开关
                            context=context_enc,     # 上下文大小
                            empty=last_freq,         # 是否为空编码器
                            **kwt                    # 时域特定参数
                )
                self.tencoder.append(tenc)

            if multi:
                enc = MultiWrap(enc, multi_freqs)
            self.encoder.append(enc)
            if index == 0:
                chin = self.audio_channels * len(self.sources)
                chin_z = chin
                if self.cac:
                    chin_z *= 2
            dec = HDecLayer(
                chout_z,
                chin_z,
                dconv=dconv_mode & 2,
                last=index == 0,
                context=context,
                **kw_dec
            )
            if multi:
                dec = MultiWrap(dec, multi_freqs)
            if freq:
                tdec = HDecLayer(
                    chout,
                    chin,
                    dconv=dconv_mode & 2,
                    empty=last_freq,
                    last=index == 0,
                    context=context,
                    **kwt
                )
                self.tdecoder.insert(0, tdec)
            self.decoder.insert(0, dec)

            chin = chout
            chin_z = chout_z
            chout = int(growth * chout)
            chout_z = int(growth * chout_z)
            if freq:
                if freqs <= kernel_size:
                    freqs = 1
                else:
                    freqs //= stride
            if index == 0 and freq_emb:
                self.freq_emb = ScaledEmbedding(
                    freqs, chin_z, smooth=emb_smooth, scale=emb_scale
                )
                self.freq_emb_scale = freq_emb

        if rescale:
            rescale_module(self, reference=rescale)

        transformer_channels = channels * growth ** (depth - 1)
        if bottom_channels:
            self.channel_upsampler = nn.Conv1d(transformer_channels, bottom_channels, 1)
            self.channel_downsampler = nn.Conv1d(
                bottom_channels, transformer_channels, 1
            )
            self.channel_upsampler_t = nn.Conv1d(
                transformer_channels, bottom_channels, 1
            )
            self.channel_downsampler_t = nn.Conv1d(
                bottom_channels, transformer_channels, 1
            )

            transformer_channels = bottom_channels

        if t_layers > 0:
            self.crosstransformer = CrossTransformerEncoder(
                dim=transformer_channels,
                emb=t_emb,
                hidden_scale=t_hidden_scale,
                num_heads=t_heads,
                num_layers=t_layers,
                cross_first=t_cross_first,
                dropout=t_dropout,
                max_positions=t_max_positions,
                norm_in=t_norm_in,
                norm_in_group=t_norm_in_group,
                group_norm=t_group_norm,
                norm_first=t_norm_first,
                norm_out=t_norm_out,
                max_period=t_max_period,
                weight_decay=t_weight_decay,
                lr=t_lr,
                layer_scale=t_layer_scale,
                gelu=t_gelu,
                sin_random_shift=t_sin_random_shift,
                weight_pos_embed=t_weight_pos_embed,
                cape_mean_normalize=t_cape_mean_normalize,
                cape_augment=t_cape_augment,
                cape_glob_loc_scale=t_cape_glob_loc_scale,
                sparse_self_attn=t_sparse_self_attn,
                sparse_cross_attn=t_sparse_cross_attn,
                mask_type=t_mask_type,
                mask_random_seed=t_mask_random_seed,
                sparse_attn_window=t_sparse_attn_window,
                global_window=t_global_window,
                sparsity=t_sparsity,
                auto_sparsity=t_auto_sparsity,
            )
        else:
            self.crosstransformer = None

    def _spec(self, x):
        hl = self.hop_length
        nfft = self.nfft
        x0 = x  # noqa

        # We re-pad the signal in order to keep the property
        # that the size of the output is exactly the size of the input
        # divided by the stride (here hop_length), when divisible.
        # This is achieved by padding by 1/4th of the kernel size (here nfft).
        # which is not supported by torch.stft.
        # Having all convolution operations follow this convention allow to easily
        # align the time and frequency branches later on.
        assert hl == nfft // 4
        le = int(math.ceil(x.shape[-1] / hl))
        pad = hl // 2 * 3
        x = pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect")

        z = spectro(x, nfft, hl)[..., :-1, :]
        assert z.shape[-1] == le + 4, (z.shape, x.shape, le)
        z = z[..., 2: 2 + le]
        return z

    def _ispec(self, z, length=None, scale=0):
        hl = self.hop_length // (4**scale)
        z = F.pad(z, (0, 0, 0, 1))
        z = F.pad(z, (2, 2))
        pad = hl // 2 * 3
        le = hl * int(math.ceil(length / hl)) + 2 * pad
        x = ispectro(z, hl, length=le)
        x = x[..., pad: pad + length]
        return x

    def _magnitude(self, z):
        # return the magnitude of the spectrogram, except when cac is True,
        # in which case we just move the complex dimension to the channel one.
        if self.cac:
            B, C, Fr, T = z.shape
            m = torch.view_as_real(z).permute(0, 1, 4, 2, 3)
            m = m.reshape(B, C * 2, Fr, T)
        else:
            m = z.abs()
        return m

    def _mask(self, z, m):
        # Apply masking given the mixture spectrogram `z` and the estimated mask `m`.
        # If `cac` is True, `m` is actually a full spectrogram and `z` is ignored.
        niters = self.wiener_iters
        if self.cac:
            B, S, C, Fr, T = m.shape
            out = m.view(B, S, -1, 2, Fr, T).permute(0, 1, 2, 4, 5, 3)
            out = torch.view_as_complex(out.contiguous())
            return out
        if self.training:
            niters = self.end_iters
        if niters < 0:
            z = z[:, None]
            return z / (1e-8 + z.abs()) * m
        else:
            return self._wiener(m, z, niters)

    def _wiener(self, mag_out, mix_stft, niters):
        # apply wiener filtering from OpenUnmix.
        init = mix_stft.dtype
        wiener_win_len = 300
        residual = self.wiener_residual

        B, S, C, Fq, T = mag_out.shape
        mag_out = mag_out.permute(0, 4, 3, 2, 1)
        mix_stft = torch.view_as_real(mix_stft.permute(0, 3, 2, 1))

        outs = []
        for sample in range(B):
            pos = 0
            out = []
            for pos in range(0, T, wiener_win_len):
                frame = slice(pos, pos + wiener_win_len)
                z_out = wiener(
                    mag_out[sample, frame],
                    mix_stft[sample, frame],
                    niters,
                    residual=residual,
                )
                out.append(z_out.transpose(-1, -2))
            outs.append(torch.cat(out, dim=0))
        out = torch.view_as_complex(torch.stack(outs, 0))
        out = out.permute(0, 4, 3, 2, 1).contiguous()
        if residual:
            out = out[:, :-1]
        assert list(out.shape) == [B, S, C, Fq, T]
        return out.to(init)

    def valid_length(self, length: int):
        """
        Return a length that is appropriate for evaluation.
        In our case, always return the training length, unless
        it is smaller than the given length, in which case this
        raises an error.
        """
        if not self.use_train_segment:
            return length
        training_length = int(self.segment * self.samplerate)
        if training_length < length:
            raise ValueError(
                    f"Given length {length} is longer than "
                    f"training length {training_length}")
        return training_length

    def forward(self, mix):
        """
        HTDemucs模型的前向传播函数
        
        Args:
            mix: 输入的混合音频，形状为 (batch_size, channels, time_length)
                 例如: (4, 2, 441000) 表示4个样本，2通道，10秒音频
        
        Returns:
            分离后的音源，形状为 (batch_size, num_sources, channels, time_length)
            例如: (4, 4, 2, 441000) 表示4个样本，4个分离源，2通道，10秒音频
        
        详细流程：
        1. 输入预处理：长度调整和填充
        2. 频域变换：STFT变换获取频谱
        3. 双分支标准化：分别对频域和时域进行标准化
        4. 并行编码：频域和时域编码器同时处理
        5. 跨域融合：Transformer实现时频域信息交互
        6. 并行解码：频域和时域解码器重建特征
        7. 后处理：掩码应用、逆STFT、分支融合
        """
        # =============================================================================
        # 第一阶段：输入预处理和长度管理
        # =============================================================================
        
        # 获取输入音频的原始长度（采样点数）
        length = mix.shape[-1]  
        
        # 初始化预填充长度记录，用于后续裁剪回原始长度
        length_pre_pad = None
        
        # 训练段长度处理：确保输入长度符合训练要求
        if self.use_train_segment:
            if self.training:
                # 训练模式：动态调整段长度为当前输入长度
                self.segment = Fraction(mix.shape[-1], self.samplerate)
            else:
                # 推理模式：使用固定的训练段长度
                training_length = int(self.segment * self.samplerate)  # 例如：10*44100=441000
                if mix.shape[-1] < training_length:
                    # 如果输入长度小于训练长度，需要填充
                    length_pre_pad = mix.shape[-1]  # 记录原始长度
                    # 右侧填零到训练长度
                    mix = F.pad(mix, (0, training_length - length_pre_pad))
        
        # =============================================================================
        # 第二阶段：频域变换和预处理
        # =============================================================================
        
        # STFT变换：将时域信号转换为频域复数谱
        z = self._spec(mix)  # (B, C, Freq, Time)
        # mix.shape: torch.Size([4, 1, 396900])
        # z.shape: torch.Size([4, 1, 1024, 776])
        
        # 提取幅度谱或复数通道表示
        mag = self._magnitude(z).to(mix.device)  # 频域幅度谱，形状同z
        
        # 频域分支的输入特征
        x = mag  # x将作为频域分支的输入
        
        # 获取频域特征的维度信息
        B, C, Fq, T = x.shape  # B:批次大小, C:通道数, Fq:频率bins, T:时间帧数
        
        # =============================================================================
        # 第三阶段：双分支标准化处理
        # =============================================================================
        
        # 频域分支标准化：对整个频谱进行归一化
        # 计算全局均值（跨通道、频率、时间维度）
        mean = x.mean(dim=(1, 2, 3), keepdim=True)  # 形状：(B, 1, 1, 1)
        # 计算全局标准差
        std = x.std(dim=(1, 2, 3), keepdim=True)   # 形状：(B, 1, 1, 1)
        # Z-score标准化
        x = (x - mean) / (1e-5 + std)  # 防止除零，1e-5为数值稳定性常数
        # x现在是频域分支的标准化输入
        
        # 时域分支预处理：对原始波形进行标准化
        xt = mix  # 时域分支直接使用原始混合音频
        # 计算时域信号的均值（跨通道和时间维度）
        meant = xt.mean(dim=(1, 2), keepdim=True)  # 形状：(B, 1, 1)
        # 计算时域信号的标准差
        stdt = xt.std(dim=(1, 2), keepdim=True)    # 形状：(B, 1, 1)
        # 时域Z-score标准化
        xt = (xt - meant) / (1e-5 + stdt)
        
        # =============================================================================
        # 第四阶段：编码器阶段 - 并行特征提取
        # =============================================================================
        
        # 初始化跳跃连接和长度记录列表
        saved = []      # 频域分支的跳跃连接特征
        saved_t = []    # 时域分支的跳跃连接特征  
        lengths = []    # 频域分支各层的长度信息，用于解码时恢复
        lengths_t = []  # 时域分支各层的长度信息
        
        # 遍历所有编码器层，进行多尺度特征提取
        for idx, encode in enumerate(self.encoder):
            # 记录当前层频域特征的时间长度
            lengths.append(x.shape[-1])
            
            # 初始化注入特征（用于分支合并）
            inject = None
            
            # 时域编码器处理（并行于频域编码器）
            if idx < len(self.tencoder):
                # 还没有到达分支合并点，继续并行处理
                
                # 记录时域特征长度
                lengths_t.append(xt.shape[-1])
                
                # 获取对应的时域编码器
                tenc = self.tencoder[idx]
                
                # 时域编码器前向传播
                xt = tenc(xt)
                
                if not tenc.empty:
                    # 非空编码器，保存特征用于跳跃连接
                    saved_t.append(xt)
                else:
                    # 空编码器（只包含第一个卷积），表示到达分支合并点
                    # 此时时域和频域分支具有相同形状，可以进行融合
                    inject = xt  # 将时域特征注入频域分支
            
            # 频域编码器前向传播（可能包含时域特征注入）
            x = encode(x, inject)
            
            # 频率嵌入：在第一层之后添加频率位置编码
            if idx == 0 and self.freq_emb is not None:
                # 生成频率索引
                frs = torch.arange(x.shape[-2], device=x.device)  # [0, 1, 2, ..., Fq-1]
                
                # 计算频率嵌入并扩展到匹配特征维度
                emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                
                # 将频率嵌入加权加到特征上
                x = x + self.freq_emb_scale * emb
            
            # 保存当前层特征用于解码器的跳跃连接
            saved.append(x)
        
        # =============================================================================
        # 第五阶段：Transformer跨域融合
        # =============================================================================
        
        if self.crosstransformer:
            # 通道维度调整（如果需要）
            if self.bottom_channels:
                # 保存原始形状信息
                b, c, f, t = x.shape
                
                # 频域特征：重排维度并上采样通道数
                x = rearrange(x, "b c f t-> b c (f t)")  # 展平空间维度
                x = self.channel_upsampler(x)             # 1x1卷积调整通道数
                x = rearrange(x, "b c (f t)-> b c f t", f=f)  # 恢复空间维度
                
                # 时域特征：上采样通道数
                xt = self.channel_upsampler_t(xt)
            
            # 跨域Transformer：实现时频域信息交互
            # x: 频域特征, xt: 时域特征
            # 输出：融合后的频域和时域特征
            x, xt = self.crosstransformer(x, xt)
            
            # 通道维度恢复（如果之前进行了调整）
            if self.bottom_channels:
                # 频域特征：下采样通道数
                x = rearrange(x, "b c f t-> b c (f t)")
                x = self.channel_downsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                
                # 时域特征：下采样通道数  
                xt = self.channel_downsampler_t(xt)
        
        # =============================================================================
        # 第六阶段：解码器阶段 - 特征重建和上采样
        # =============================================================================
        
        # 遍历所有解码器层，进行特征重建
        for idx, decode in enumerate(self.decoder):
            # 获取对应编码器层的跳跃连接特征（LIFO顺序）
            skip = saved.pop(-1)
            
            # 频域解码器前向传播
            # skip: 跳跃连接特征, lengths.pop(-1): 目标长度
            # 返回: x: 解码特征, pre: 最终转置卷积前的特征
            x, pre = decode(x, skip, lengths.pop(-1))
            
            # 时域解码器处理（与频域解码器同步）
            offset = self.depth - len(self.tdecoder)  # 计算时域解码器开始的偏移
            
            if idx >= offset:
                # 到达时域解码器处理范围
                tdec = self.tdecoder[idx - offset]  # 获取对应的时域解码器
                length_t = lengths_t.pop(-1)       # 获取目标长度
                
                if tdec.empty:
                    # 空解码器：分支分离点，使用频域的pre特征
                    assert pre.shape[2] == 1, pre.shape  # 确保频率维度为1
                    pre = pre[:, :, 0]                    # 移除频率维度
                    xt, _ = tdec(pre, None, length_t)     # 时域解码
                else:
                    # 非空解码器：使用时域跳跃连接
                    skip = saved_t.pop(-1)               # 获取时域跳跃连接
                    xt, _ = tdec(xt, skip, length_t)      # 时域解码
        
        # 验证所有跳跃连接都已使用完毕
        assert len(saved) == 0     # 频域跳跃连接全部使用
        assert len(lengths_t) == 0 # 时域长度信息全部使用
        assert len(saved_t) == 0   # 时域跳跃连接全部使用
        
        # =============================================================================
        # 第七阶段：输出后处理和分支融合
        # =============================================================================
        
        # 获取音源数量
        S = len(self.sources)  # 2个源['dry', 'rir']
        
        # 重塑频域特征为多源输出格式
        x = x.view(B, S, -1, Fq, T)  # (batch, sources, channels, freq, time)
        
        # 频域反标准化：恢复到原始量纲
        x = x * std[:, None] + mean[:, None]  # 使用之前保存的均值和标准差
        
        # MPS设备兼容性处理（Apple Silicon GPU）
        # 因为MPS不支持复数运算，需要临时转到CPU
        x_is_mps = x.device.type == "mps"
        if x_is_mps:
            x = x.cpu()
        
        # 应用掩码或维纳滤波，获取分离后的复数频谱
        zout = self._mask(z, x)  # z: 原始复数频谱, x: 估计的掩码/频谱
        
        # 逆STFT：将频域复数频谱转换回时域波形
        if self.use_train_segment:
            if self.training:
                # 训练模式：使用原始长度
                x = self._ispec(zout, length)
            else:
                # 推理模式：使用训练长度
                x = self._ispec(zout, training_length)
        else:
            # 不使用训练段模式：直接使用原始长度
            x = self._ispec(zout, length)
        
        # 恢复MPS设备
        if x_is_mps:
            x = x.to("mps")
        
        # 时域分支处理：重塑为多源格式
        if self.use_train_segment:
            if self.training:
                # 训练模式：使用原始长度重塑
                xt = xt.view(B, S, -1, length)
            else:
                # 推理模式：使用训练长度重塑
                xt = xt.view(B, S, -1, training_length)
        else:
            # 不使用训练段模式
            xt = xt.view(B, S, -1, length)
        
        # 时域反标准化：恢复到原始量纲
        xt = xt * stdt[:, None] + meant[:, None]
        
        # 双分支融合：频域重建 + 时域重建
        x = xt + x  # 最终输出 = 时域输出 + 频域输出
        
        # 裁剪到原始长度（如果之前进行了填充）
        if length_pre_pad:
            x = x[..., :length_pre_pad]
        
        # 返回分离后的音源
        # 输出形状：(batch_size, num_sources, channels, time_length)
        return x
