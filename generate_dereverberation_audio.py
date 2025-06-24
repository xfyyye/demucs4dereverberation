#!/usr/bin/env python3
"""
去混响音频生成脚本
专门用于从混响音频中分离出干声(dry)和房间冲击响应(rir)
"""

import torch
import torchaudio as ta
from pathlib import Path
import argparse
import sys

# 添加demucs路径
sys.path.append('demucs')

from demucs.apply import apply_model

def load_model(model_path):
    """加载训练好的HTDemucs模型"""
    print(f"正在加载模型: {model_path}")
    
    # 从检查点加载
    package = torch.load(model_path, map_location='cpu')
    
    # 检查是否是新格式的模型文件
    if 'klass' in package and 'kwargs' in package:
        # 新格式：直接使用klass和kwargs重建模型
        model_class = package['klass']
        model_kwargs = package['kwargs']
        model = model_class(**model_kwargs)
        model.load_state_dict(package['state'])
    else:
        # 旧格式：从args重建
        if 'best_state' in package:
            model_state = package['best_state']
        else:
            model_state = package['state']
        
        args = package['args']
        from demucs.htdemucs import HTDemucs
        
        # 获取sources配置
        sources = list(args.dset.sources)
        
        model = HTDemucs(
            sources=sources, 
            audio_channels=args.dset.channels,
            samplerate=args.dset.samplerate,
            **args.htdemucs
        )
        model.load_state_dict(model_state)
    
    model.eval()
    print(f"模型加载成功，支持的音源: {model.sources}")
    return model

def separate_audio(model, mixture_path, output_dir, device='cpu'):
    """
    分离单个音频文件
    
    Args:
        model: 训练好的HTDemucs模型
        mixture_path: 混响音频文件路径
        output_dir: 输出目录
        device: 计算设备
    """
    print(f"正在处理: {mixture_path}")
    
    # 加载音频
    mixture, sr = ta.load(mixture_path)
    print(f"音频采样率: {sr} Hz, 形状: {mixture.shape}")
    
    # 确保音频格式正确: (channels, time)
    if mixture.dim() == 1:
        mixture = mixture.unsqueeze(0)  # 单声道: (1, time)
    
    # 为模型添加batch维度: (1, channels, time)
    mixture = mixture.unsqueeze(0).to(device)
    model = model.to(device)
    
    # 进行音源分离
    print("正在进行音源分离...")
    with torch.no_grad():
        sources = apply_model(model, mixture, split=True, overlap=0.25)
    
    # sources形状: [batch, num_sources, channels, time]
    print(f"分离结果形状: {sources.shape}")
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 获取文件名（不含扩展名）
    input_name = Path(mixture_path).stem
    
    # 保存分离的音源
    source_names = model.sources
    for i, source_name in enumerate(source_names):
        # 提取对应的音源: [channels, time]
        source_audio = sources[0, i]  # 去掉batch维度，保留[channels, time]
        
        # 保存音频文件
        save_path = output_dir / f"{input_name}_{source_name}.wav"
        ta.save(save_path, source_audio.cpu(), sr)
        print(f"已保存 {source_name}: {save_path}")
    
    # 如果有干声，额外保存一个去混响版本
    if 'dry' in source_names:
        dry_idx = source_names.index('dry')
        dry_audio = sources[0, dry_idx]  # [channels, time]
        
        save_path = output_dir / f"{input_name}_dereverberated.wav"
        ta.save(save_path, dry_audio.cpu(), sr)
        print(f"已保存去混响音频: {save_path}")

def process_test_dataset(model, test_dir, output_dir, num_samples=None, device='cpu'):
    """
    处理测试数据集中的样本
    
    Args:
        model: 训练好的模型
        test_dir: 测试数据集目录
        output_dir: 输出目录
        num_samples: 处理的样本数量，None表示处理全部
        device: 计算设备
    """
    test_path = Path(test_dir)
    output_path = Path(output_dir)
    
    if not test_path.exists():
        print(f"错误: 测试目录不存在: {test_path}")
        return
    
    # 查找所有样本目录
    sample_dirs = sorted(list(test_path.glob('sample*')))
    
    if not sample_dirs:
        print(f"错误: 在 {test_path} 中没有找到样本目录")
        return
    
    # 限制处理的样本数量
    if num_samples is not None:
        sample_dirs = sample_dirs[:num_samples]
    
    print(f"找到 {len(sample_dirs)} 个样本，开始处理...")
    
    success_count = 0
    for i, sample_dir in enumerate(sample_dirs, 1):
        print(f"\n--- 处理样本 {i}/{len(sample_dirs)}: {sample_dir.name} ---")
        
        # 查找mixture.wav文件
        mixture_file = sample_dir / 'mixture.wav'
        if not mixture_file.exists():
            print(f"警告: 未找到 {mixture_file}，跳过此样本")
            continue
        
        # 为每个样本创建输出目录
        sample_output_dir = output_path / sample_dir.name
        
        try:
            # 进行音源分离
            separate_audio(model, mixture_file, sample_output_dir, device)
            success_count += 1
            
            print(f"✓ 样本 {sample_dir.name} 处理完成")
            
        except Exception as e:
            print(f"✗ 处理样本 {sample_dir.name} 时出错: {e}")
            continue
    
    print(f"\n=== 处理完成 ===")
    print(f"成功处理: {success_count}/{len(sample_dirs)} 个样本")
    print(f"结果保存在: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='HTDemucs去混响音频生成工具')
    parser.add_argument('--model', required=True, help='模型检查点文件路径(.th)')
    parser.add_argument('--test_dir', required=True, help='测试数据集目录路径')
    parser.add_argument('--output', required=True, help='输出目录路径')
    parser.add_argument('--num_samples', type=int, default=None, 
                       help='处理的样本数量，默认处理全部')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], 
                       help='计算设备 (默认: cpu)')
    
    args = parser.parse_args()
    
    # 检查CUDA可用性
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU")
        args.device = 'cpu'
    
    print("=== HTDemucs 去混响工具 ===")
    print(f"模型文件: {args.model}")
    print(f"测试目录: {args.test_dir}")
    print(f"输出目录: {args.output}")
    print(f"处理设备: {args.device}")
    if args.num_samples:
        print(f"处理样本数: {args.num_samples}")
    else:
        print("处理样本数: 全部")
    
    # 加载模型
    try:
        model = load_model(args.model)
    except Exception as e:
        print(f"错误: 无法加载模型 - {e}")
        return
    
    # 处理测试数据集
    try:
        process_test_dataset(
            model=model,
            test_dir=args.test_dir,
            output_dir=args.output,
            num_samples=args.num_samples,
            device=args.device
        )
    except Exception as e:
        print(f"错误: 处理过程中出现问题 - {e}")
        return
    
    print("\n=== 去混响处理完成 ===")
    print("生成的文件说明:")
    print("- *_dry.wav: 分离出的干声(去混响后的音频)")
    print("- *_rir.wav: 分离出的房间冲击响应")
    print("- *_dereverberated.wav: 干声的副本(方便识别)")

if __name__ == "__main__":
    main() 