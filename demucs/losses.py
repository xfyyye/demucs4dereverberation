# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
去混响专用损失函数模块

提供多种适用于去混响任务的损失函数，包括：
- 时域、频域和感知损失
- 一致性损失（确保物理约束）
- RIR正则化损失

主要类：
    DereverbLoss: 综合去混响损失函数
    
主要函数：
    fft_convolution: FFT卷积实现
    get_dereverberation_loss: 工厂函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from typing import Dict, Tuple, Optional


class DereverbLoss(nn.Module):
    """
    去混响专用综合损失函数
    
    组合以下损失:
    1. 时域重建损失 (L1/MSE/Huber)
    2. 频域幅度损失 (STFT幅度谱+相位)
    3. 感知损失 (Mel谱)
    4. 一致性损失 (pred_dry ⊛ pred_rir ≈ mix，使用真实卷积)
    5. RIR正则化损失 (稀疏性+能量衰减)
    """
    
    def __init__(
        self,
        sr: int = 16000,
        # 损失权重
        dry_weight: float = 3.0,
        rir_weight: float = 1.0,
        time_weight: float = 1.0,
        freq_weight: float = 0.5,
        mel_weight: float = 0.3,
        consistency_weight: float = 0.2,
        rir_reg_weight: float = 0.1,
        # 时域损失
        time_loss_type: str = "l1",  # "l1", "mse", "huber"
        # 卷积方法
        use_time_domain_conv: bool = True,
        # 频域设置
        n_fft: int = 1024,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        # Mel谱设置
        n_mels: int = 80,
        mel_scale: str = "htk",
        # 其他
        eps: float = 1e-8,
        device: str = 'cuda',
    ):
        super().__init__()
        
        self.sr = sr
        self.dry_weight = dry_weight
        self.rir_weight = rir_weight
        self.time_weight = time_weight
        self.freq_weight = freq_weight
        self.mel_weight = mel_weight
        self.consistency_weight = consistency_weight
        self.rir_reg_weight = rir_reg_weight
        self.time_loss_type = time_loss_type
        self.use_time_domain_conv = use_time_domain_conv
        self.eps = eps
        
        # 设置默认参数
        if hop_length is None:
            hop_length = n_fft // 4
        if win_length is None:
            win_length = n_fft
            
        # STFT变换
        self.stft_transform = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            power=None,  # 返回复数谱
            normalized=True
        )
        
        # Mel频谱变换
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            mel_scale=mel_scale,
            normalized=True
        )
        
        # 将变换器移到指定设备
        self.device = device
        self.to(device)
        
    def to(self, device):
        """移动模块到指定设备"""
        super().to(device)
        self.stft_transform = self.stft_transform.to(device)
        self.mel_transform = self.mel_transform.to(device)
        self.device = device
        return self
        
    def compute_time_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算时域损失"""
        if self.time_loss_type == "l1":
            return F.l1_loss(pred, target)
        elif self.time_loss_type == "mse":
            return F.mse_loss(pred, target)
        else:
            return F.l1_loss(pred, target)
    
    def compute_freq_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算频域损失"""
        # 计算STFT
        pred_stft = self.stft_transform(pred)
        target_stft = self.stft_transform(target)
        
        # 幅度谱损失
        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)
        mag_loss = F.l1_loss(pred_mag, target_mag)
        
        # 相位损失
        pred_phase = torch.angle(pred_stft)
        target_phase = torch.angle(target_stft)
        phase_loss = 1 - F.cosine_similarity(
            torch.stack([torch.cos(pred_phase), torch.sin(pred_phase)], dim=-1),
            torch.stack([torch.cos(target_phase), torch.sin(target_phase)], dim=-1),
            dim=-1
        ).mean()
        
        return mag_loss + 0.1 * phase_loss
    
    def compute_mel_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算Mel感知损失"""
        # 计算Mel频谱
        pred_mel = self.mel_transform(pred)
        target_mel = self.mel_transform(target)
        
        # 对数Mel频谱更符合人耳感知
        pred_log_mel = torch.log(pred_mel + self.eps)
        target_log_mel = torch.log(target_mel + self.eps)
        
        return F.l1_loss(pred_log_mel, target_log_mel)
    
    
    def compute_consistency_loss(self, pred_dry: torch.Tensor, pred_rir: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
        """
        计算一致性损失: pred_dry ⊛ pred_rir ≈ mix
        使用与数据生成一致的卷积方法
        """
        if self.use_time_domain_conv:
            return self._time_domain_convolution_original(pred_dry, pred_rir, mix)
        else:
            return fft_convolution(pred_dry, pred_rir, mix)
    
    def _time_domain_convolution_original(self, pred_dry: torch.Tensor, pred_rir: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
        """
        原始时域卷积实现，与数据生成过程（scipy.signal.convolve）完全一致
        修正了设备支持问题
        """
        batch_size, channels, length = pred_dry.shape
        device = pred_dry.device
        
        # 对每个样本进行真正的卷积操作
        reconstructed_list = []
        
        for b in range(batch_size):
            for c in range(channels):
                dry_single = pred_dry[b, c]  # [length]
                rir_single = pred_rir[b, c]  # [length]
                
                # 实现真正的卷积：y[n] = sum(x[m] * h[n-m])
                reconstructed_single = torch.zeros(length, device=device)  # 修正：指定设备
                for n in range(length):
                    for m in range(length):
                        if n - m >= 0 and n - m < length:
                            reconstructed_single[n] += dry_single[m] * rir_single[n - m]
                
                reconstructed_list.append(reconstructed_single)
        
        # 重新组合为原始形状
        reconstructed = torch.stack(reconstructed_list).view(batch_size, channels, length)
        
        # 确保长度一致
        min_length = min(reconstructed.shape[-1], mix.shape[-1])
        reconstructed = reconstructed[..., :min_length]
        mix_truncated = mix[..., :min_length]
        
        return F.l1_loss(reconstructed, mix_truncated)
    
    def compute_rir_regularization(self, pred_rir: torch.Tensor) -> torch.Tensor:
        """
        计算RIR正则化损失
        鼓励RIR具有合理的时域特性：
        1. 稀疏性（大部分值应该接近0）
        2. 能量衰减（随时间递减）
        """
        # 1. 稀疏性约束：L1正则化
        sparsity_loss = torch.mean(torch.abs(pred_rir))
        
        # 2. 能量衰减约束
        length = pred_rir.shape[-1]
        segment_length = length // 4
        total_decay_loss = 0
        
        for i in range(3):
            start1, end1 = i * segment_length, (i + 1) * segment_length
            start2, end2 = (i + 1) * segment_length, (i + 2) * segment_length
            
            if end2 <= length:
                energy1 = torch.mean(pred_rir[:, :, start1:end1] ** 2)
                energy2 = torch.mean(pred_rir[:, :, start2:end2] ** 2)
                # 鼓励能量递减
                decay_loss = F.relu(energy2 - 0.8 * energy1)
                total_decay_loss += decay_loss
        
        return sparsity_loss + total_decay_loss
    
    def forward(
        self, 
        estimate: torch.Tensor, 
        sources: torch.Tensor, 
        mix: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算总损失
        
        Args:
            estimate: 模型预测 [batch, 2, channels, time] (dry, rir)
            sources: 真实标签 [batch, 2, channels, time] (dry, rir)
            mix: 混响音频 [batch, channels, time]
            
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的详细信息
        """
        pred_dry = estimate[:, 0]    # [batch, channels, time]
        pred_rir = estimate[:, 1]    # [batch, channels, time]
        target_dry = sources[:, 0]   # [batch, channels, time]
        target_rir = sources[:, 1]   # [batch, channels, time]
        
        loss_dict = {}
        
        # 1. 干声重建损失
        dry_time_loss = self.compute_time_loss(pred_dry, target_dry)
        dry_freq_loss = self.compute_freq_loss(pred_dry, target_dry)
        dry_mel_loss = self.compute_mel_loss(pred_dry, target_dry)
        
        total_dry_loss = (
            self.time_weight * dry_time_loss +
            self.freq_weight * dry_freq_loss +
            self.mel_weight * dry_mel_loss
        )
        
        # 2. RIR重建损失
        rir_time_loss = self.compute_time_loss(pred_rir, target_rir)
        rir_reg_loss = self.compute_rir_regularization(pred_rir)
        
        total_rir_loss = rir_time_loss + self.rir_reg_weight * rir_reg_loss
        
        # 3. 一致性损失
        consistency_loss = self.compute_consistency_loss(pred_dry, pred_rir, mix)
        
        # 4. 总损失（干声权重更高）
        total_loss = (
            self.dry_weight * total_dry_loss +
            self.rir_weight * total_rir_loss +
            self.consistency_weight * consistency_loss
        )
        
        # 记录各项损失
        loss_dict.update({
            'dry_time_loss': dry_time_loss,
            'dry_freq_loss': dry_freq_loss,
            'dry_mel_loss': dry_mel_loss,
            'rir_time_loss': rir_time_loss,
            'rir_reg_loss': rir_reg_loss,
            'consistency_loss': consistency_loss,
            'total_dry_loss': total_dry_loss,
            'total_rir_loss': total_rir_loss,
        })
        
        return total_loss, loss_dict





def fft_convolution(pred_dry: torch.Tensor, pred_rir: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    """
    FFT卷积实现（计算更快）
    
    Args:
        pred_dry: 预测的干声 [batch, channels, time]
        pred_rir: 预测的RIR [batch, channels, time]
        mix: 混响音频 [batch, channels, time]
    
    Returns:
        一致性损失值
    """
    batch_size, channels, length = pred_dry.shape
    
    # 使用FFT实现快速卷积
    # 为避免循环卷积的边界效应，填充到2*length-1
    fft_size = 2 * length - 1
    
    # FFT变换
    dry_fft = torch.fft.rfft(pred_dry, n=fft_size, dim=-1)
    rir_fft = torch.fft.rfft(pred_rir, n=fft_size, dim=-1)
    
    # 频域乘法 = 时域卷积
    conv_fft = dry_fft * rir_fft
    
    # 逆FFT得到卷积结果
    reconstructed = torch.fft.irfft(conv_fft, n=fft_size, dim=-1)
    
    # 截取到原始长度
    reconstructed = reconstructed[..., :length]
    
    # 确保长度一致
    min_length = min(reconstructed.shape[-1], mix.shape[-1])
    reconstructed = reconstructed[..., :min_length]
    mix_truncated = mix[..., :min_length]
    
    return F.l1_loss(reconstructed, mix_truncated)


def get_dereverberation_loss(config: dict, device: str = 'cuda') -> DereverbLoss:
    """
    Args:
        config: 损失函数配置字典
        device: 设备类型
    
    Returns:
        配置好的DereverbLoss实例
    """
    # 过滤有效参数
    valid_params = {
        'sr', 'dry_weight', 'rir_weight', 'time_weight', 'freq_weight', 'mel_weight', 
        'consistency_weight', 'rir_reg_weight', 'time_loss_type', 'use_time_domain_conv',
        'n_fft', 'hop_length', 'win_length', 'n_mels', 'mel_scale', 'eps'
    }
    
    filtered_config = {k: v for k, v in config.items() if k in valid_params}
    filtered_config['device'] = device
    
    return DereverbLoss(**filtered_config) 