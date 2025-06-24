#!/usr/bin/env python3
"""
去混响专用评估指标
包含适合去混响任务的各种音频质量评估指标
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy import signal
from scipy.stats import pearsonr
import librosa

def sdr_loss(reference, estimate, eps=1e-8):
    """
    计算信号失真比 (Signal-to-Distortion Ratio)
    适用于去混响任务的核心指标
    """
    num = torch.sum(reference**2, dim=-1, keepdim=True)
    den = torch.sum((reference - estimate)**2, dim=-1, keepdim=True)
    return 10 * torch.log10((num + eps) / (den + eps))

def pesq_estimate(reference, estimate, sr=16000):
    """
    估算PESQ分数 (简化版本)
    PESQ是语音质量的重要指标，对去混响很有意义
    """
    # 简化的PESQ估算，基于频谱距离
    ref_stft = torch.stft(reference, n_fft=512, return_complex=True)
    est_stft = torch.stft(estimate, n_fft=512, return_complex=True)
    
    ref_mag = torch.abs(ref_stft)
    est_mag = torch.abs(est_stft)
    
    # 计算频谱距离
    spectral_distance = torch.mean((ref_mag - est_mag)**2)
    # 转换为类PESQ分数 (1-5)
    pesq_score = 5.0 - torch.clamp(spectral_distance * 100, 0, 4)
    return pesq_score

def stoi_estimate(reference, estimate, sr=16000):
    """
    估算STOI分数 (简化版本)
    短时客观可懂度指数，对去混响后的语音质量很重要
    """
    # 确保输入是二维的: [channels, time] 或 [time]
    if reference.dim() == 1:
        reference = reference.unsqueeze(0)
    if estimate.dim() == 1:
        estimate = estimate.unsqueeze(0)
    
    # 如果是多声道，取平均
    if reference.shape[0] > 1:
        reference = reference.mean(0)
    if estimate.shape[0] > 1:
        estimate = estimate.mean(0)
    
    # 简化的STOI估算，基于短时段的相关性
    win_len = int(0.025 * sr)  # 25ms窗口
    hop_len = int(0.010 * sr)  # 10ms跳跃
    
    # 确保信号长度足够
    min_length = min(reference.shape[-1], estimate.shape[-1])
    if min_length < win_len:
        win_len = min_length // 2
        hop_len = win_len // 2
    
    reference = reference[:min_length]
    estimate = estimate[:min_length]
    
    # 手动进行分帧
    correlations = []
    for start in range(0, min_length - win_len + 1, hop_len):
        end = start + win_len
        ref_frame = reference[start:end]
        est_frame = estimate[start:end]
        
        # 计算相关性（避免零向量）
        if torch.norm(ref_frame) > 1e-8 and torch.norm(est_frame) > 1e-8:
            correlation = F.cosine_similarity(ref_frame, est_frame, dim=0)
            correlations.append(correlation)
    
    if len(correlations) == 0:
        return torch.tensor(0.0)
    
    # 返回平均相关性，映射到[0,1]区间
    avg_correlation = torch.mean(torch.stack(correlations))
    return torch.clamp((avg_correlation + 1) / 2, 0, 1)  # 从[-1,1]映射到[0,1]

def reverberation_time_error(reference, estimate, sr=16000):
    """
    混响时间误差
    比较参考信号和估计信号的混响衰减特性
    """
    def estimate_rt60(signal, sr):
        # 计算信号的能量衰减曲线
        energy = torch.cumsum(torch.flip(signal**2, dims=[-1]), dim=-1)
        energy = torch.flip(energy, dims=[-1])
        
        # 找到-60dB衰减点
        max_energy = torch.max(energy)
        target_energy = max_energy * 10**(-60/10)
        
        # 简化的RT60估算
        decay_idx = torch.argmax((energy < target_energy).float())
        rt60 = decay_idx.float() / sr
        return rt60
    
    ref_rt60 = estimate_rt60(reference, sr)
    est_rt60 = estimate_rt60(estimate, sr)
    
    return torch.abs(ref_rt60 - est_rt60)

def spectral_centroid_error(reference, estimate):
    """
    频谱重心误差
    衡量频谱特性的保持程度
    """
    def spectral_centroid(signal):
        stft = torch.stft(signal, n_fft=1024, return_complex=True)
        magnitude = torch.abs(stft)
        
        freqs = torch.linspace(0, 1, magnitude.shape[1])
        freqs = freqs.to(signal.device).unsqueeze(0).unsqueeze(-1)
        
        centroid = torch.sum(magnitude * freqs, dim=1) / (torch.sum(magnitude, dim=1) + 1e-8)
        return torch.mean(centroid)
    
    ref_centroid = spectral_centroid(reference)
    est_centroid = spectral_centroid(estimate)
    
    return torch.abs(ref_centroid - est_centroid)

def comprehensive_dereverberation_metrics(reference_dry, estimated_dry, 
                                        reference_rir=None, estimated_rir=None,
                                        sr=16000):
    """
    综合去混响评估指标
    
    Args:
        reference_dry: 参考干声
        estimated_dry: 估计干声  
        reference_rir: 参考RIR (可选)
        estimated_rir: 估计RIR (可选)
        sr: 采样率
    
    Returns:
        dict: 包含各种评估指标的字典
    """
    metrics = {}
    
    # 基础指标
    metrics['sdr_dry'] = sdr_loss(reference_dry, estimated_dry).item()
    metrics['pesq_estimate'] = pesq_estimate(reference_dry, estimated_dry, sr).item()
    metrics['stoi_estimate'] = stoi_estimate(reference_dry, estimated_dry, sr).item()
    
    # 频谱特性
    metrics['spectral_centroid_error'] = spectral_centroid_error(reference_dry, estimated_dry).item()
    
    # 混响特性
    metrics['rt60_error'] = reverberation_time_error(reference_dry, estimated_dry, sr).item()
    
    # 如果提供了RIR，也评估RIR的重建质量
    if reference_rir is not None and estimated_rir is not None:
        metrics['sdr_rir'] = sdr_loss(reference_rir, estimated_rir).item()
    
    # 计算整体评分 (加权平均)
    weights = {
        'sdr_dry': 0.4,
        'pesq_estimate': 0.3,
        'stoi_estimate': 0.2,
        'spectral_centroid_error': -0.05,  # 负权重，误差越小越好
        'rt60_error': -0.05
    }
    
    overall_score = sum(weights[k] * metrics[k] for k in weights if k in metrics)
    metrics['overall_score'] = overall_score
    
    return metrics

def batch_evaluate_dereverberation(reference_batch, estimated_batch, sr=16000):
    """
    批量评估去混响效果
    
    Args:
        reference_batch: [batch, channels, time] 参考信号
        estimated_batch: [batch, channels, time] 估计信号
    
    Returns:
        dict: 平均指标
    """
    batch_size = reference_batch.shape[0]
    all_metrics = []
    
    for i in range(batch_size):
        ref = reference_batch[i].mean(0) if reference_batch.dim() == 3 else reference_batch[i]
        est = estimated_batch[i].mean(0) if estimated_batch.dim() == 3 else estimated_batch[i]
        
        metrics = comprehensive_dereverberation_metrics(ref, est, sr=sr)
        all_metrics.append(metrics)
    
    # 计算平均值
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    return avg_metrics 