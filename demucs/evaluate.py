# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Test time evaluation, either using the original SDR from [Vincent et al. 2006]
or the newest SDR definition from the MDX 2021 competition (this one will
be reported as `nsdr` for `new sdr`).
"""

from concurrent import futures
import logging
import os
from pathlib import Path

from dora.log import LogProgress
import numpy as np
import musdb
import museval
import torch as th
import torchaudio
import torch.nn.functional as F

from .apply import apply_model
from .audio import convert_audio, save_audio
from . import distrib
from .utils import DummyPoolExecutor


logger = logging.getLogger(__name__)


class CustomTrack:
    """自定义数据集的Track类，模拟musdb.Track接口"""
    def __init__(self, name, mixture_path, targets_dict, samplerate=16000):
        self.name = name
        self.mixture_path = mixture_path
        self.targets_dict = targets_dict
        self.samplerate = samplerate
        
        # 加载mixture音频
        mixture, sr = torchaudio.load(mixture_path)
        if sr != samplerate:
            mixture = torchaudio.functional.resample(mixture, sr, samplerate)
        self.audio = mixture.numpy().T  # 转换为 (time, channels) 格式
        
        # 加载target音频
        self.targets = {}
        for name, path in targets_dict.items():
            target, sr = torchaudio.load(path)
            if sr != samplerate:
                target = torchaudio.functional.resample(target, sr, samplerate)
            
            # 创建target对象
            class Target:
                def __init__(self, audio):
                    self.audio = audio.numpy().T
            
            self.targets[name] = Target(target)


class CustomDB:
    """自定义数据集的DB类，模拟musdb.DB接口"""
    def __init__(self, root_path, sources, samplerate=16000):
        self.root_path = Path(root_path)
        self.sources = sources
        self.samplerate = samplerate
        self.tracks = []
        
        # 扫描测试目录
        test_dir = self.root_path / "test"
        if test_dir.exists():
            for sample_dir in sorted(test_dir.iterdir()):
                if sample_dir.is_dir():
                    mixture_path = sample_dir / "mixture.wav"
                    if mixture_path.exists():
                        targets_dict = {}
                        for source in sources:
                            target_path = sample_dir / f"{source}.wav"
                            if target_path.exists():
                                targets_dict[source] = target_path
                        
                        if len(targets_dict) == len(sources):
                            track = CustomTrack(
                                name=sample_dir.name,
                                mixture_path=mixture_path,
                                targets_dict=targets_dict,
                                samplerate=samplerate
                            )
                            self.tracks.append(track)
        
        logger.info(f"Loaded {len(self.tracks)} test tracks from {test_dir}")


def new_sdr(references, estimates):
    """
    Compute the SDR according to the MDX challenge definition.
    Adapted from AIcrowd/music-demixing-challenge-starter-kit (MIT license)
    """
    assert references.dim() == 4
    assert estimates.dim() == 4
    delta = 1e-7  # avoid numerical errors
    num = th.sum(th.square(references), dim=(2, 3))
    den = th.sum(th.square(references - estimates), dim=(2, 3))
    num += delta
    den += delta
    scores = 10 * th.log10(num / den)
    return scores

'''
新增评估指标函数，用于计算去混响专用指标
'''
def estimate_pesq(reference, estimate, sr=16000):
    """
    估算PESQ分数 (简化版本)
    PESQ是语音质量的重要指标，对去混响很有意义
    """
    ref_stft = th.stft(reference, n_fft=512, return_complex=True)
    est_stft = th.stft(estimate, n_fft=512, return_complex=True)
    
    ref_mag = th.abs(ref_stft)
    est_mag = th.abs(est_stft)
    
    spectral_distance = th.mean((ref_mag - est_mag)**2)
    pesq_score = 5.0 - th.clamp(spectral_distance * 100, 0, 4)
    return pesq_score.item()


def estimate_stoi(reference, estimate, sr=16000):
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
        return 0.0
        
    reference = reference[:min_length]
    estimate = estimate[:min_length]
    
    # 手动进行分帧
    correlations = []
    for start in range(0, min_length - win_len + 1, hop_len):
        end = start + win_len
        ref_frame = reference[start:end]
        est_frame = estimate[start:end]
        
        # 计算相关性（避免零向量）
        if th.norm(ref_frame) > 1e-8 and th.norm(est_frame) > 1e-8:
            correlation = F.cosine_similarity(ref_frame, est_frame, dim=0)
            correlations.append(correlation)
    
    if len(correlations) == 0:
        return 0.0
    
    # 返回平均相关性，映射到[0,1]区间
    avg_correlation = th.mean(th.stack(correlations))
    return th.clamp((avg_correlation + 1) / 2, 0, 1).item()


def compute_rt60_error(reference, estimate, sr=16000):
    """
    混响时间误差
    比较参考信号和估计信号的混响衰减特性
    """
    def estimate_rt60(signal):
        # 计算信号的能量衰减曲线
        energy = th.cumsum(th.flip(signal**2, dims=[-1]), dim=-1)
        energy = th.flip(energy, dims=[-1])
        
        # 找到-60dB衰减点
        max_energy = th.max(energy)
        target_energy = max_energy * 10**(-60/10)
        
        # 简化的RT60估算
        decay_idx = th.argmax((energy < target_energy).float())
        rt60 = decay_idx.float() / sr
        return rt60
    
    ref_rt60 = estimate_rt60(reference)
    est_rt60 = estimate_rt60(estimate)
    
    return th.abs(ref_rt60 - est_rt60).item()


def compute_dereverberation_metrics(references, estimates, sources_names, sr=16000, config=None):
    """
    计算去混响专用评估指标
    
    Args:
        references: 真实标签 [batch, num_sources, channels, time]
        estimates: 模型预测 [batch, num_sources, channels, time] 
        sources_names: 源名称列表，如 ['dry', 'rir']
        sr: 采样率
        config: 配置字典，指定计算哪些指标
    
    Returns:
        dict: 包含各种评估指标的字典
    """
    if config is None:
        config = {
            'compute_pesq': True,
            'compute_stoi': True,
            'compute_rt60': True,
        }
    
    metrics = {}
    
    # 只对 'dry' 源计算语音质量指标
    if 'dry' in sources_names:
        dry_idx = sources_names.index('dry')
        reference_dry = references[:, dry_idx]  # [batch, channels, time]
        estimated_dry = estimates[:, dry_idx]   # [batch, channels, time]
        
        batch_size = reference_dry.shape[0]
        
        # 计算各项指标的累积值
        sdr_scores = []
        pesq_scores = []
        stoi_scores = []
        rt60_errors = []
        
        for i in range(batch_size):
            # 处理每个样本，确保统一为1维[time]
            ref_sample = reference_dry[i]  # [channels, time]
            est_sample = estimated_dry[i]  # [channels, time]
            
            # 如果是多声道，取平均得到[time]
            if ref_sample.dim() > 1 and ref_sample.shape[0] > 1:
                ref = ref_sample.mean(0)  # [channels, time] → [time]
                est = est_sample.mean(0)  # [channels, time] → [time]
            else:
                # 单声道，去掉channel维度
                ref = ref_sample.squeeze(0)  # [1, time] → [time]
                est = est_sample.squeeze(0)  # [1, time] → [time]
            
            # SDR for dry signal - 确保4维：[batch, sources, channels, time]
            ref_expanded = ref.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [time] → [1, 1, 1, time]
            est_expanded = est.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [time] → [1, 1, 1, time]
            # ref_expanded.shape torch.Size([1, 1, 1, 441000])
            # est_expanded.shape torch.Size([1, 1, 1, 441000])
            sdr = new_sdr(ref_expanded, est_expanded)
            sdr_scores.append(sdr.item())
            
            # PESQ估算
            if config.get('compute_pesq', True):
                pesq_score = estimate_pesq(ref, est, sr)
                pesq_scores.append(pesq_score)
            
            # STOI估算  
            if config.get('compute_stoi', True):
                stoi_score = estimate_stoi(ref, est, sr)
                stoi_scores.append(stoi_score)
            
            # 混响时间误差
            if config.get('compute_rt60', True):
                rt60_error = compute_rt60_error(ref, est, sr)
                rt60_errors.append(rt60_error)
        
        # 计算平均值
        if sdr_scores:
            metrics['sdr_dry_dereverb'] = sum(sdr_scores) / len(sdr_scores)
        if pesq_scores:
            metrics['pesq_estimate'] = sum(pesq_scores) / len(pesq_scores)
        if stoi_scores:
            metrics['stoi_estimate'] = sum(stoi_scores) / len(stoi_scores)
        if rt60_errors:
            metrics['rt60_error'] = sum(rt60_errors) / len(rt60_errors)
    
    return metrics
'''
--------------------------------
'''

def eval_track(references, estimates, win, hop, compute_sdr=True):
    references = references.transpose(1, 2).double()
    estimates = estimates.transpose(1, 2).double()

    new_scores = new_sdr(references.cpu()[None], estimates.cpu()[None])[0]

    if not compute_sdr:
        return None, new_scores
    else:
        references = references.numpy()
        estimates = estimates.numpy()
        scores = museval.metrics.bss_eval(
            references, estimates,
            compute_permutation=False,
            window=win,
            hop=hop,
            framewise_filters=False,
            bsseval_sources_version=False)[:-1]
        return scores, new_scores


def evaluate(solver, compute_sdr=False):
    """
    Evaluate model using museval.
    compute_sdr=False means using only the MDX definition of the SDR, which
    is much faster to evaluate.
    """

    args = solver.args

    output_dir = solver.folder / "results"
    output_dir.mkdir(exist_ok=True, parents=True)
    json_folder = solver.folder / "results/test"
    json_folder.mkdir(exist_ok=True, parents=True)

    # 加载测试数据集
    if hasattr(args.dset, 'wav') and args.dset.wav:
        # 使用自定义WAV数据集
        test_set = CustomDB(
            root_path=args.dset.wav,
            sources=args.dset.sources,
            samplerate=args.dset.samplerate
        )
        src_rate = args.dset.samplerate
    else:
        logger.warning("No test dataset available, skipping evaluation")
        return {}

    if len(test_set.tracks) == 0:
        logger.warning("No test tracks found, skipping evaluation")
        return {}

    eval_device = 'cpu'

    model = solver.model
    win = int(1. * model.samplerate)
    hop = int(1. * model.samplerate)

    indexes = range(distrib.rank, len(test_set.tracks), distrib.world_size)
    indexes = LogProgress(logger, indexes, updates=args.misc.num_prints,
                          name='Eval')
    pendings = []

    # 在分布式训练中禁用多进程评估，避免死锁
    pool = DummyPoolExecutor
    with pool(0) as pool:
        for index in indexes:
            track = test_set.tracks[index]

            mix = th.from_numpy(track.audio).t().float()
            if mix.dim() == 1:
                mix = mix[None]
            mix = mix.to(solver.device)
            ref = mix.mean(dim=0)  # mono mixture
            mix = (mix - ref.mean()) / ref.std()
            mix = convert_audio(mix, src_rate, model.samplerate, model.audio_channels)
            estimates = apply_model(model, mix[None],
                                    shifts=args.test.shifts, split=args.test.split,
                                    overlap=args.test.overlap)[0]
            estimates = estimates * ref.std() + ref.mean()
            estimates = estimates.to(eval_device)

            references = th.stack(
                [th.from_numpy(track.targets[name].audio).t() for name in model.sources])
            if references.dim() == 2:
                references = references[:, None]
            references = references.to(eval_device)
            references = convert_audio(references, src_rate,
                                       model.samplerate, model.audio_channels)
            if args.test.save:
                folder = solver.folder / "wav" / track.name
                folder.mkdir(exist_ok=True, parents=True)
                for name, estimate in zip(model.sources, estimates):
                    save_audio(estimate.cpu(), folder / (name + ".mp3"), model.samplerate)

            pendings.append((track.name, pool.submit(
                eval_track, references, estimates, win=win, hop=hop, compute_sdr=compute_sdr)))

        pendings = LogProgress(logger, pendings, updates=args.misc.num_prints,
                               name='Eval (BSS)')
        tracks = {}
        for track_name, pending in pendings:
            pending = pending.result()
            scores, nsdrs = pending
            tracks[track_name] = {}
            for idx, target in enumerate(model.sources):
                tracks[track_name][target] = {'nsdr': [float(nsdrs[idx])]}
            if scores is not None:
                (sdr, isr, sir, sar) = scores
                for idx, target in enumerate(model.sources):
                    values = {
                        "SDR": sdr[idx].tolist(),
                        "SIR": sir[idx].tolist(),
                        "ISR": isr[idx].tolist(),
                        "SAR": sar[idx].tolist()
                    }
                    tracks[track_name][target].update(values)
            #-----改动-----
            # 计算去混响专用指标
            if hasattr(args, 'dereverberation_metrics') and args.dereverberation_metrics.get('enable', False):
                # 为计算去混响指标准备数据
                references_batch = references.unsqueeze(0)  # [1, num_sources, channels, time]
                estimates_batch = estimates.unsqueeze(0)    # [1, num_sources, channels, time]
                
                dereverb_metrics = compute_dereverberation_metrics(
                    references_batch, 
                    estimates_batch, 
                    model.sources,
                    sr=model.samplerate,
                    config=args.dereverberation_metrics
                )
                
                # 将去混响指标添加到track结果中
                if not hasattr(tracks[track_name], 'dereverberation'):
                    tracks[track_name]['dereverberation'] = {}
                tracks[track_name]['dereverberation'].update(dereverb_metrics)
            #-----改动-----

        all_tracks = {}
        for src in range(distrib.world_size):
            all_tracks.update(distrib.share(tracks, src))

        result = {}
        if all_tracks:
            metric_names = next(iter(all_tracks.values()))[model.sources[0]]
        for metric_name in metric_names:
            avg = 0
            avg_of_medians = 0
            for source in model.sources:
                medians = [
                    np.nanmedian(all_tracks[track][source][metric_name])
                    for track in all_tracks.keys()]
                mean = np.mean(medians)
                median = np.median(medians)
                result[metric_name.lower() + "_" + source] = mean
                result[metric_name.lower() + "_med" + "_" + source] = median
                avg += mean / len(model.sources)
                avg_of_medians += median / len(model.sources)
            result[metric_name.lower()] = avg
            result[metric_name.lower() + "_med"] = avg_of_medians
        #-----改动-----
        # 添加去混响专用指标的汇总
        if all_tracks and hasattr(args, 'dereverberation_metrics') and args.dereverberation_metrics.get('enable', False):
            # 检查是否有去混响指标
            first_track = next(iter(all_tracks.values()))
            if 'dereverberation' in first_track:
                dereverb_metrics_names = first_track['dereverberation'].keys()
                for metric_name in dereverb_metrics_names:
                    # 收集所有track的该指标值
                    values = [
                        all_tracks[track]['dereverberation'][metric_name]
                        for track in all_tracks.keys()
                        if 'dereverberation' in all_tracks[track] and metric_name in all_tracks[track]['dereverberation']
                    ]
                    if values:
                        result[f"dereverb_{metric_name}"] = np.mean(values)
                        result[f"dereverb_{metric_name}_med"] = np.median(values)
        #-----改动-----
        return result
