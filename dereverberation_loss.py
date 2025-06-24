#!/usr/bin/env python3
"""
去混响任务专用损失函数
结合时域、频域和感知损失，优化去混响效果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from typing import Dict, Tuple, Optional


class DereverbLoss(nn.Module):
    """
    去混响专用复合损失函数
    
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
        # 频域设置
        n_fft: int = 1024,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        # Mel谱设置
        n_mels: int = 80,
        mel_scale: str = "htk",
        # 其他
        eps: float = 1e-8,
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
        self.eps = eps
        
        # 设置默认参数
        if hop_length is None:
            hop_length = n_fft // 4
        if win_length is None:
            win_length = n_fft
            
        # STFT变换
        self.stft = T.Spectrogram(
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
        
        # 确保变换器在正确的设备上
        self.device = None
        
    def time_domain_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """时域重建损失"""
        if self.time_loss_type == "l1":
            return F.l1_loss(pred, target)
        elif self.time_loss_type == "mse":
            return F.mse_loss(pred, target)
        elif self.time_loss_type == "huber":
            return F.huber_loss(pred, target, delta=0.1)
        else:
            raise ValueError(f"Unsupported time loss type: {self.time_loss_type}")
    
    def frequency_domain_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """频域幅度损失"""
        # 确保变换器在正确的设备上
        if self.device != pred.device:
            self.stft = self.stft.to(pred.device)
            self.device = pred.device
            
        # 计算STFT
        pred_stft = self.stft(pred)  # [batch, freq, time]
        target_stft = self.stft(target)
        
        # 计算幅度谱
        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)
        
        # L1损失更适合幅度谱
        mag_loss = F.l1_loss(pred_mag, target_mag)
        
        # 可选：添加相位损失（对音质有帮助）
        pred_phase = torch.angle(pred_stft)
        target_phase = torch.angle(target_stft)
        
        # 使用余弦相似度计算相位损失
        phase_loss = 1 - F.cosine_similarity(
            torch.stack([torch.cos(pred_phase), torch.sin(pred_phase)], dim=-1),
            torch.stack([torch.cos(target_phase), torch.sin(target_phase)], dim=-1),
            dim=-1
        ).mean()
        
        return mag_loss + 0.1 * phase_loss
    
    def mel_perceptual_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mel频谱感知损失"""
        # 确保变换器在正确的设备上
        if self.device != pred.device:
            self.mel_transform = self.mel_transform.to(pred.device)
            self.device = pred.device
            
        # 计算Mel频谱
        pred_mel = self.mel_transform(pred)  # [batch, n_mels, time]
        target_mel = self.mel_transform(target)
        
        # 对数Mel频谱更符合人耳感知
        pred_log_mel = torch.log(pred_mel + self.eps)
        target_log_mel = torch.log(target_mel + self.eps)
        
        return F.l1_loss(pred_log_mel, target_log_mel)
    
    def consistency_loss(
        self, 
        pred_dry: torch.Tensor, 
        pred_rir: torch.Tensor, 
        mix: torch.Tensor
    ) -> torch.Tensor:
        """
        一致性损失：确保 pred_dry ⊛ pred_rir ≈ mix
        使用正确的卷积操作，与数据生成过程（scipy.signal.convolve）一致
        """
        batch_size, channels, length = pred_dry.shape
        
        # 对每个样本进行真正的卷积操作
        reconstructed_list = []
        
        for b in range(batch_size):
            for c in range(channels):
                dry_single = pred_dry[b, c]  # [length]
                rir_single = pred_rir[b, c]  # [length]
                
                # 实现真正的卷积：y[n] = sum(x[m] * h[n-m])
                reconstructed_single = torch.zeros(length)
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
    
    def rir_regularization_loss(self, pred_rir: torch.Tensor) -> torch.Tensor:
        """
        RIR正则化损失
        鼓励RIR具有合理的时域特性：
        1. 初始冲击响应应该较强
        2. 随时间衰减
        3. 稀疏性（大部分值应该接近0）
        """
        batch_size, channels, length = pred_rir.shape
        
        # 1. 稀疏性约束：L1正则化
        sparsity_loss = torch.mean(torch.abs(pred_rir))
        
        # 2. 能量衰减约束：后面的部分应该比前面的能量小
        # 将RIR分成几段，后面的段能量应该递减
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
        predictions: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor],
        mix: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算总损失
        
        Args:
            predictions: {'dry': pred_dry, 'rir': pred_rir}
            targets: {'dry': target_dry, 'rir': target_rir}
            mix: 原始混响音频
            
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的详细信息
        """
        pred_dry = predictions['dry']
        pred_rir = predictions['rir'] 
        target_dry = targets['dry']
        target_rir = targets['rir']
        
        loss_dict = {}
        
        # 1. 干声重建损失
        dry_time_loss = self.time_domain_loss(pred_dry, target_dry)
        dry_freq_loss = self.frequency_domain_loss(pred_dry, target_dry)
        dry_mel_loss = self.mel_perceptual_loss(pred_dry, target_dry)
        
        total_dry_loss = (
            self.time_weight * dry_time_loss +
            self.freq_weight * dry_freq_loss +
            self.mel_weight * dry_mel_loss
        )
        
        # 2. RIR重建损失（主要使用时域损失）
        rir_time_loss = self.time_domain_loss(pred_rir, target_rir)
        rir_reg_loss = self.rir_regularization_loss(pred_rir)
        
        total_rir_loss = rir_time_loss + self.rir_reg_weight * rir_reg_loss
        
        # 3. 一致性损失
        consist_loss = self.consistency_loss(pred_dry, pred_rir, mix)
        
        # 4. 总损失（干声权重更高）
        total_loss = (
            self.dry_weight * total_dry_loss +  # 干声是主要目标
            self.rir_weight * total_rir_loss +  # RIR作为辅助
            self.consistency_weight * consist_loss
        )
        
        # 记录各项损失
        loss_dict.update({
            'total_loss': total_loss.item(),
            'dry_time_loss': dry_time_loss.item(),
            'dry_freq_loss': dry_freq_loss.item(),
            'dry_mel_loss': dry_mel_loss.item(),
            'rir_time_loss': rir_time_loss.item(),
            'rir_reg_loss': rir_reg_loss.item(),
            'consistency_loss': consist_loss.item(),
            'total_dry_loss': total_dry_loss.item(),
            'total_rir_loss': total_rir_loss.item(),
        })
        
        return total_loss, loss_dict





# 工厂函数
def get_dereverberation_loss(**kwargs):
    """
    获取去混响损失函数
    
    Args:
        **kwargs: DereverbLoss的参数
    """
    # 过滤掉DereverbLoss不接受的参数
    valid_params = {
        'sr', 'dry_weight', 'rir_weight', 'time_weight', 'freq_weight', 'mel_weight', 
        'consistency_weight', 'rir_reg_weight', 'time_loss_type',
        'n_fft', 'hop_length', 'win_length', 'n_mels', 'mel_scale', 'eps'
    }
    
    # 移除特殊参数（用于选择损失类型等）
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    
    # 处理特殊的参数映射
    if 'loss_type' in kwargs:
        # loss_type参数不传给DereverbLoss，这里只是为了兼容配置
        pass
    
    return DereverbLoss(**filtered_kwargs) 