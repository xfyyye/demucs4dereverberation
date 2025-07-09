# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""主训练循环模块,包含训练器类和相关工具函数"""

import logging

from dora import get_xp
from dora.utils import write_and_rename
from dora.log import LogProgress, bold
import torch
import torch.nn.functional as F

from . import augment, distrib, states, pretrained
from .apply import apply_model
from .ema import ModelEMA
from .evaluate import evaluate, new_sdr
from .svd import svd_penalty
from .utils import pull_metric, EMA

logger = logging.getLogger(__name__)


def _summary(metrics):
    """格式化输出指标摘要"""
    return " | ".join(f"{key.capitalize()}={val}" for key, val in metrics.items())


class Solver(object):
    """
    训练器类,负责模型训练、验证和测试的主要逻辑
    
    主要功能:
    - 模型训练和优化
    - 模型验证和测试
    - 模型状态保存和加载
    - 指标记录和输出
    """
    def __init__(self, loaders, model, optimizer, args):
        """
        初始化训练器
        
        Args:
            loaders: 数据加载器字典,包含训练和验证数据集
            model: 待训练的模型
            optimizer: 优化器
            args: 配置参数
        """
        self.args = args
        self.loaders = loaders

        self.model = model
        self.optimizer = optimizer
        # 初始化量化器
        self.quantizer = states.get_quantizer(self.model, args.quant, self.optimizer)
        # 分布式包装模型
        self.dmodel = distrib.wrap(model)
        self.device = next(iter(self.model.parameters())).device

        # 初始化指数移动平均(EMA)模型
        # 分为每批次更新和每轮更新两种
        self.emas = {'batch': [], 'epoch': []}
        for kind in self.emas.keys():
            decays = getattr(args.ema, kind)
            device = self.device if kind == 'batch' else 'cpu'
            if decays:
                for decay in decays:
                    self.emas[kind].append(ModelEMA(self.model, decay, device=device))

        # 初始化数据增强
        augments = [augment.Shift(shift=int(args.dset.samplerate * args.dset.shift),
                                  same=args.augment.shift_same)]
        if args.augment.flip:
            augments += [augment.FlipChannels(), augment.FlipSign()]
        for aug in ['scale', 'remix']:
            kw = getattr(args.augment, aug)
            if kw.proba:
                augments.append(getattr(augment, aug.capitalize())(**kw))
        self.augment = torch.nn.Sequential(*augments)

        # 设置模型保存路径
        xp = get_xp()
        self.folder = xp.folder
        self.checkpoint_file = xp.folder / 'checkpoint.th'
        self.best_file = xp.folder / 'best.th'
        logger.debug("Checkpoint will be saved to %s", self.checkpoint_file.resolve())
        self.best_state = None
        self.best_changed = False

        self.link = xp.link
        self.history = self.link.history

        #-----改动-----
        # 初始化去混响专用损失函数
        self._init_dereverberation_loss()
        #-----改动-----
        # 重置训练状态
        self._reset()

    def _serialize(self, epoch):
        """
        序列化模型状态
        
        保存:
        - 模型参数
        - 优化器状态
        - 训练历史
        - 最佳模型状态
        - EMA模型状态
        """
        package = {}
        package['state'] = self.model.state_dict()
        package['optimizer'] = self.optimizer.state_dict()
        package['history'] = self.history
        package['best_state'] = self.best_state
        package['args'] = self.args
        for kind, emas in self.emas.items():
            for k, ema in enumerate(emas):
                package[f'ema_{kind}_{k}'] = ema.state_dict()
        with write_and_rename(self.checkpoint_file) as tmp:
            torch.save(package, tmp)

        # 定期保存检查点
        save_every = self.args.save_every
        if save_every and (epoch + 1) % save_every == 0 and epoch + 1 != self.args.epochs:
            with write_and_rename(self.folder / f'checkpoint_{epoch + 1}.th') as tmp:
                torch.save(package, tmp)

        # 保存最佳模型
        if self.best_changed:
            with write_and_rename(self.best_file) as tmp:
                package = states.serialize_model(self.model, self.args)
                package['state'] = self.best_state
                torch.save(package, tmp)
            self.best_changed = False

    def _reset(self):
        """
        重置训练器状态
        
        - 从检查点恢复
        - 加载预训练模型
        - 从指定模型继续训练
        """
        if self.checkpoint_file.exists():
            logger.info(f'Loading checkpoint model: {self.checkpoint_file}')
            package = torch.load(self.checkpoint_file, 'cpu')
            self.model.load_state_dict(package['state'])
            self.optimizer.load_state_dict(package['optimizer'])
            self.history[:] = package['history']
            self.best_state = package['best_state']
            for kind, emas in self.emas.items():
                for k, ema in enumerate(emas):
                    ema.load_state_dict(package[f'ema_{kind}_{k}'])
        elif self.args.continue_pretrained:
            model = pretrained.get_model(
                name=self.args.continue_pretrained,
                repo=self.args.pretrained_repo)
            self.model.load_state_dict(model.state_dict())
        elif self.args.continue_from:
            name = 'checkpoint.th'
            root = self.folder.parent
            cf = root / str(self.args.continue_from) / name
            logger.info("Loading from %s", cf)
            package = torch.load(cf, 'cpu')
            self.best_state = package['best_state']
            if self.args.continue_best:
                self.model.load_state_dict(package['best_state'], strict=False)
            else:
                self.model.load_state_dict(package['state'], strict=False)
            if self.args.continue_opt:
                self.optimizer.load_state_dict(package['optimizer'])

    def _init_dereverberation_loss(self):
        """初始化去混响专用损失函数"""
        self.use_dereverb_loss = getattr(self.args, 'dereverberation_loss', {}).get('enable', False)
        
        if self.use_dereverb_loss:
            from .losses import get_dereverberation_loss
            
            config = self.args.dereverberation_loss
            config['sr'] = self.args.dset.samplerate  # 添加采样率
            
            # 创建去混响损失函数实例
            self.dereverb_loss_fn = get_dereverberation_loss(config, device=self.device)
            self.dereverb_loss_fn.to(self.device)
            


    def _format_train(self, metrics: dict) -> dict:
        """
        格式化训练/验证指标
        
        Args:
            metrics: 原始指标字典
        Returns:
            格式化后的指标字典
        """
        losses = {
            'loss': format(metrics['loss'], ".4f"),
            'reco': format(metrics['reco'], ".4f"),
        }
        if 'nsdr' in metrics:
            losses['nsdr'] = format(metrics['nsdr'], ".3f")
        if self.quantizer is not None:
            losses['ms'] = format(metrics['ms'], ".2f")
        if 'grad' in metrics:
            losses['grad'] = format(metrics['grad'], ".4f")
        if 'best' in metrics:
            losses['best'] = format(metrics['best'], '.4f')
        if 'bname' in metrics:
            losses['bname'] = metrics['bname']
        if 'penalty' in metrics:
            losses['penalty'] = format(metrics['penalty'], ".4f")
        if 'hloss' in metrics:
            losses['hloss'] = format(metrics['hloss'], ".4f")
            
        # 添加去混响专用指标格式化
        if 'sdr_dry_dereverb' in metrics:
            losses['sdr_dry'] = format(metrics['sdr_dry_dereverb'], ".3f")
        if 'pesq_estimate' in metrics:
            losses['pesq'] = format(metrics['pesq_estimate'], ".3f")
        if 'stoi_estimate' in metrics:
            losses['stoi'] = format(metrics['stoi_estimate'], ".3f")
        if 'rt60_error' in metrics:
            losses['rt60_err'] = format(metrics['rt60_error'], ".4f")
            
        return losses

    def _format_test(self, metrics: dict) -> dict:
        """
        格式化测试指标
        
        Args:
            metrics: 原始指标字典
        Returns:
            格式化后的指标字典
        """
        losses = {}
        if 'sdr' in metrics:
            losses['sdr'] = format(metrics['sdr'], '.3f')
        if 'nsdr' in metrics:
            losses['nsdr'] = format(metrics['nsdr'], '.3f')
        for source in self.model.sources:
            key = f'sdr_{source}'
            if key in metrics:
                losses[key] = format(metrics[key], '.3f')
            key = f'nsdr_{source}'
            if key in metrics:
                losses[key] = format(metrics[key], '.3f')
        return losses

    def train(self):
        """
        主训练循环
        
        - 训练每个epoch
        - 进行验证
        - 保存检查点
        - 评估测试集
        """
        # 重放历史指标
        if self.history:
            logger.info("Replaying metrics from previous run")
        for epoch, metrics in enumerate(self.history):
            formatted = self._format_train(metrics['train'])
            logger.info(
                bold(f'Train Summary | Epoch {epoch + 1} | {_summary(formatted)}'))
            formatted = self._format_train(metrics['valid'])
            logger.info(
                bold(f'Valid Summary | Epoch {epoch + 1} | {_summary(formatted)}'))
            if 'test' in metrics:
                formatted = self._format_test(metrics['test'])
                if formatted:
                    logger.info(bold(f"Test Summary | Epoch {epoch + 1} | {_summary(formatted)}"))

        epoch = 0
        for epoch in range(len(self.history), self.args.epochs):
            # 训练一个epoch
            self.model.train()  # 开启BatchNorm和Dropout
            metrics = {}
            logger.info('-' * 70)
            logger.info("Training...")
            metrics['train'] = self._run_one_epoch(epoch)
            formatted = self._format_train(metrics['train'])
            logger.info(
                bold(f'Train Summary | Epoch {epoch + 1} | {_summary(formatted)}'))

            # 交叉验证
            logger.info('-' * 70)
            logger.info('Cross validation...')
            self.model.eval()  # 关闭BatchNorm和Dropout
            with torch.no_grad():
                valid = self._run_one_epoch(epoch, train=False)
                bvalid = valid
                bname = 'main'
                state = states.copy_state(self.model.state_dict())
                metrics['valid'] = {}
                metrics['valid']['main'] = valid
                key = self.args.test.metric
                # 评估所有EMA模型
                for kind, emas in self.emas.items():
                    for k, ema in enumerate(emas):
                        with ema.swap():
                            valid = self._run_one_epoch(epoch, train=False)
                        name = f'ema_{kind}_{k}'
                        metrics['valid'][name] = valid
                        a = valid[key]
                        b = bvalid[key]
                        if key.startswith('nsdr'):
                            a = -a
                            b = -b
                        if a < b:
                            bvalid = valid
                            state = ema.state
                            bname = name
                    metrics['valid'].update(bvalid)
                    metrics['valid']['bname'] = bname

            # 更新最佳模型
            valid_loss = metrics['valid'][key]
            mets = pull_metric(self.link.history, f'valid.{key}') + [valid_loss]
            if key.startswith('nsdr'):
                best_loss = max(mets)
            else:
                best_loss = min(mets)
            metrics['valid']['best'] = best_loss
            if self.args.svd.penalty > 0:
                kw = dict(self.args.svd)
                kw.pop('penalty')
                with torch.no_grad():
                    penalty = svd_penalty(self.model, exact=True, **kw)
                metrics['valid']['penalty'] = penalty

            formatted = self._format_train(metrics['valid'])
            logger.info(
                bold(f'Valid Summary | Epoch {epoch + 1} | {_summary(formatted)}'))

            # 保存最佳模型
            if valid_loss == best_loss or self.args.dset.train_valid:
                logger.info(bold('New best valid loss %.4f'), valid_loss)
                self.best_state = states.copy_state(state)
                self.best_changed = True

            # 每隔test.every个epoch或最后一个epoch评估模型
            should_eval = (epoch + 1) % self.args.test.every == 0
            is_last = epoch == self.args.epochs - 1
            if should_eval or is_last:
                # 在测试集上评估
                logger.info('-' * 70)
                logger.info('Evaluating on the test set...')
                # 切换到最佳模型进行测试
                if self.args.test.best:
                    state = self.best_state
                else:
                    state = states.copy_state(self.model.state_dict())
                compute_sdr = self.args.test.sdr and is_last
                with states.swap_state(self.model, state):
                    with torch.no_grad():
                        metrics['test'] = evaluate(self, compute_sdr=compute_sdr)
                formatted = self._format_test(metrics['test'])
                logger.info(bold(f"Test Summary | Epoch {epoch + 1} | {_summary(formatted)}"))
            self.link.push_metrics(metrics)

            # 保存检查点
            if distrib.rank == 0:
                self._serialize(epoch)
                logger.debug("Checkpoint saved to %s", self.checkpoint_file.resolve())
            if is_last:
                break

    def _run_one_epoch(self, epoch, train=True):
        """
        运行一个epoch的训练或验证
        
        Args:
            epoch: 当前epoch
            train: 是否为训练模式
            
        Returns:
            平均损失和指标
        """
        args = self.args
        data_loader = self.loaders['train'] if train else self.loaders['valid']
        # 分布式训练时设置epoch
        if distrib.world_size > 1 and train:
            data_loader.sampler.set_epoch(epoch)

        label = ["Valid", "Train"][train]
        name = label + f" | Epoch {epoch + 1}"
        # 计算总批次数
        total = len(data_loader)
        if args.max_batches:
            total = min(total, args.max_batches)
        # 初始化进度条
        logprog = LogProgress(logger, data_loader, total=total,
                              updates=self.args.misc.num_prints, name=name)
        # 初始化指数移动平均
        averager = EMA()

        for idx, sources in enumerate(logprog):
            sources = sources.to(self.device)
            if train:
                # sources = self.augment(sources)
                # mix = sources.sum(dim=1)
                '''
                改为：
                # 训练时使用预存的mixture文件（正确的卷积合成）
                # 数据格式：[mixture, dry, rir], sources原本就是[batch, 3, channels, time]
                # 直接对原始sources增强（包含mixture）
                '''
                combined = self.augment(sources)  
                mix = combined[:, 0]
                sources = combined[:, 1:]
                # 训练时：sources.shape torch.Size([4, 2, 1, 396900]) -> [batch, num_sources, channels, time]
                # 训练时：mix.shape torch.Size([4, 1, 396900]) -> [batch, channels, time]
            else:
                # 验证时分离混合音频和源音频
                mix = sources[:, 0]
                sources = sources[:, 1:]
                # 验证时mix.shape torch.Size([1, 1, 441000])
                # 验证时sources.shape torch.Size([1, 2, 1, 441000])

            # 前向传播
            if not train and self.args.valid_apply:
                # 验证时使用特定的模型应用方法
                estimate = apply_model(self.model, mix, split=self.args.test.split, overlap=0)
            else:
                # 正常前向传播
                estimate = self.dmodel(mix)
            if train and hasattr(self.model, 'transform_target'):
                # 如果需要,转换目标
                print("transform_target")
                sources = self.model.transform_target(mix, sources)
            # 检查输出形状
            assert estimate.shape == sources.shape, (estimate.shape, sources.shape)
            dims = tuple(range(2, sources.dim()))
            # estimate.shape torch.Size([4, 2, 2, 441000]) 分布式训练，2个gpu，每个gpu 4个batch
            # sources.shape torch.Size([4, 2, 2, 441000])
            # dims = (2, 3)
            
            # 计算损失
            dereverb_loss_dict = {}
            
            if self.use_dereverb_loss and train:
                # 使用去混响专用损失
                loss, dereverb_loss_dict = self.dereverb_loss_fn(estimate, sources, mix)
                # 为了兼容原有代码，从详细损失中提取重建损失作为reco
                reco = torch.stack([
                    dereverb_loss_dict['total_dry_loss'], 
                    dereverb_loss_dict['total_rir_loss']
                ])
            else:
                # 使用原有的简单损失
                if args.optim.loss == 'l1':
                    # L1损失
                    loss = F.l1_loss(estimate, sources, reduction='none')
                    # 第一步：沿channels和time维度平均 # dims=(2,3) 第二步：沿batch维度平均  
                    loss = loss.mean(dims).mean(0)
                    reco = loss # 结果 shape: (num_sources,)  每个源的平均损失
                elif args.optim.loss == 'mse':
                    # MSE损失
                    loss = F.mse_loss(estimate, sources, reduction='none')
                    loss = loss.mean(dims)
                    reco = loss**0.5
                    reco = reco.mean(0)
                else:
                    raise ValueError(f"Invalid loss {self.args.loss}")
                # 应用权重
                weights = torch.tensor(args.weights).to(sources)
                loss = (loss * weights).sum() / weights.sum()
            # 计算量化损失
            ms = 0
            if self.quantizer is not None:
                ms = self.quantizer.model_size()
            if args.quant.diffq:
                loss += args.quant.diffq * ms

            # 记录损失
            losses = {}
            if self.use_dereverb_loss and train:
                weights = torch.tensor(args.weights).to(sources)
                losses['reco'] = (reco * weights).sum() / weights.sum()
                # 添加去混响专用损失记录
                for key, value in dereverb_loss_dict.items():
                    losses[key] = value
            else:
                weights = torch.tensor(args.weights).to(sources) if 'weights' not in locals() else weights
                losses['reco'] = (reco * weights).sum() / weights.sum()
            losses['ms'] = ms
            

            # 验证时计算NSDR和去混响专用指标
            if not train:
                nsdrs = new_sdr(sources, estimate.detach()).mean(0)
                total = 0
                for source, nsdr, w in zip(self.model.sources, nsdrs, weights):
                    losses[f'nsdr_{source}'] = nsdr
                    total += w * nsdr
                losses['nsdr'] = total / weights.sum()
                
                # 计算去混响专用指标（只在验证时计算简单指标，详细指标在evaluate.py中）
                if hasattr(self.args, 'dereverberation_metrics') and self.args.dereverberation_metrics.get('enable', False):
                    from .evaluate import compute_dereverberation_metrics
                    dereverb_metrics = compute_dereverberation_metrics(
                        sources, estimate.detach(), 
                        self.model.sources, sr=self.args.dset.samplerate,
                        config=self.args.dereverberation_metrics
                    )
                    losses.update(dereverb_metrics)

            # 计算SVD惩罚项
            if train and args.svd.penalty > 0:
                kw = dict(args.svd)
                kw.pop('penalty')
                penalty = svd_penalty(self.model, **kw)
                losses['penalty'] = penalty
                loss += args.svd.penalty * penalty

            # 记录总损失
            losses['loss'] = loss

            # 记录每个源的重建损失
            for k, source in enumerate(self.model.sources):
                losses[f'reco_{source}'] = reco[k]

            # 训练模式下的优化步骤
            if train:
                # 反向传播
                loss.backward()
                # 计算梯度范数
                grad_norm = 0
                grads = []
                for p in self.model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm()**2
                        grads.append(p.grad.data)
                losses['grad'] = grad_norm ** 0.5
                # 梯度裁剪
                if args.optim.clip_grad:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        args.optim.clip_grad)

                # 调试信息
                if self.args.flag == 'uns':
                    for n, p in self.model.named_parameters():
                        if p.grad is None:
                            print('no grad', n)
                # 优化器步骤
                self.optimizer.step()
                self.optimizer.zero_grad()
                # 更新批次EMA
                for ema in self.emas['batch']:
                    ema.update()
            # 更新平均损失
            losses = averager(losses)
            # 格式化并记录日志
            logs = self._format_train(losses)
            logprog.update(**logs)
            # 清理内存
            del loss, estimate, reco, ms
            # 检查是否达到最大批次
            if args.max_batches == idx:
                break
            # 调试模式检查
            if self.args.debug and train:
                break
            if self.args.flag == 'debug':
                break
        # 训练模式下更新epoch EMA
        if train:
            for ema in self.emas['epoch']:
                ema.update()
        # 返回平均损失
        return distrib.average(losses, idx + 1)
