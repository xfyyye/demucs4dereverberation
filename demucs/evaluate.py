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
            
            # 创建一个简单的target对象
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
        return result
