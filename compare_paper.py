"""
Transformer 与论文对比实验脚本
简化版：使用简单分词和模拟数据，聚焦于验证论文超参数配置
"""

import torch
import torch.nn as nn
import numpy as np
import os
import time
import math
from transformer import (
    Transformer,
    generate_padding_mask,
    generate_subsequent_mask,
    NoamOpt,
    LabelSmoothingLoss
)


class SyntheticTranslationDataset:
    """生成模拟翻译数据用于对比实验"""
    def __init__(self, vocab_size, seq_len, batch_size, num_samples):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.idx = 0
        
        self.src_data = []
        self.tgt_data = []
        
        for _ in range(num_samples):
            src_len = np.random.randint(5, seq_len + 1)
            tgt_len = np.random.randint(5, seq_len + 1)
            
            src_seq = torch.randint(2, vocab_size, (src_len,))
            tgt_seq = torch.cat([torch.tensor([1]), src_seq[:tgt_len - 1]])
            
            self.src_data.append(src_seq)
            self.tgt_data.append(tgt_seq)
    
    def __iter__(self):
        self.idx = 0
        return self
    
    def __next__(self):
        if self.idx >= self.num_samples:
            raise StopIteration
        
        end = min(self.idx + self.batch_size, self.num_samples)
        src_batch = self.src_data[self.idx:end]
        tgt_batch = self.tgt_data[self.idx:end]
        self.idx = end
        
        max_src_len = max(len(s) for s in src_batch)
        max_tgt_len = max(len(t) for t in tgt_batch)
        
        src_padded = torch.zeros(len(src_batch), max_src_len, dtype=torch.long)
        tgt_padded = torch.zeros(len(tgt_batch), max_tgt_len, dtype=torch.long)
        
        for i, (src, tgt) in enumerate(zip(src_batch, tgt_batch)):
            src_padded[i, :len(src)] = src
            tgt_padded[i, :len(tgt)] = tgt
        
        return src_padded, tgt_padded


def train_epoch(model, train_data, criterion, optimizer, device, pad_idx):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    
    for src, tgt in train_data:
        src = src.to(device)
        tgt = tgt.to(device)
        
        tgt_input = tgt[:, :-1]
        tgt_label = tgt[:, 1:]
        
        src_mask = generate_padding_mask(src, pad_idx=pad_idx).to(device)
        tgt_mask = generate_subsequent_mask(tgt_input.size(1)).to(device)
        
        optimizer.optimizer.zero_grad()
        output = model(src, tgt_input, src_mask, tgt_mask)
        
        output = output.contiguous().view(-1, output.size(-1))
        tgt_label = tgt_label.contiguous().view(-1)
        
        loss = criterion(output, tgt_label)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * (tgt_label != pad_idx).sum().item()
        total_tokens += (tgt_label != pad_idx).sum().item()
    
    return total_loss / total_tokens


def validate_epoch(model, val_data, criterion, device, pad_idx):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for src, tgt in val_data:
            src = src.to(device)
            tgt = tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_label = tgt[:, 1:]
            
            src_mask = generate_padding_mask(src, pad_idx=pad_idx).to(device)
            tgt_mask = generate_subsequent_mask(tgt_input.size(1)).to(device)
            
            output = model(src, tgt_input, src_mask, tgt_mask)
            
            output = output.contiguous().view(-1, output.size(-1))
            tgt_label = tgt_label.contiguous().view(-1)
            
            loss = criterion(output, tgt_label)
            
            total_loss += loss.item() * (tgt_label != pad_idx).sum().item()
            total_tokens += (tgt_label != pad_idx).sum().item()
    
    return total_loss / total_tokens


def greedy_decode(model, src, src_mask, max_len, start_token, device):
    model.eval()
    src = src.to(device)
    src_mask = src_mask.to(device)
    
    enc_output = model.encoder(src, src_mask)
    
    tgt = torch.tensor([[start_token]], dtype=torch.long).to(device)
    
    for _ in range(max_len - 1):
        tgt_mask = generate_subsequent_mask(tgt.size(1)).to(device)
        output = model.decoder(tgt, enc_output, src_mask, tgt_mask)
        
        next_token = output.argmax(dim=-1)[:, -1].unsqueeze(1)
        tgt = torch.cat([tgt, next_token], dim=1)
    
    return tgt[0].cpu().numpy()


def calculate_accuracy(model, data, device, pad_idx):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for src, tgt in data:
            src = src.to(device)
            tgt = tgt.to(device)
            
            src_mask = generate_padding_mask(src, pad_idx=pad_idx).to(device)
            decoded = greedy_decode(model, src[0:1], src_mask[0:1], tgt.size(1), 1, device)
            
            target_seq = tgt[0].cpu().numpy()
            target_len = (target_seq != pad_idx).sum()
            
            decoded_seq = decoded[1:]
            if len(decoded_seq) >= target_len - 1:
                decoded_seq = decoded_seq[:target_len - 1]
            
            correct += (decoded_seq == target_seq[1:1 + len(decoded_seq)]).sum()
            total += target_len - 1
    
    return correct / total


def run_experiment(config, device):
    """运行单个配置的实验"""
    vocab_size = config['vocab_size']
    pad_idx = 0
    sos_idx = 1
    
    train_data = SyntheticTranslationDataset(
        vocab_size=vocab_size,
        seq_len=config['max_len'],
        batch_size=config['batch_size'],
        num_samples=config['train_samples']
    )
    val_data = SyntheticTranslationDataset(
        vocab_size=vocab_size,
        seq_len=config['max_len'],
        batch_size=config['batch_size'],
        num_samples=config['val_samples']
    )
    
    model = Transformer(
        vocab_size,
        vocab_size,
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        max_len=config['max_len'],
        dropout=config['dropout']
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    
    criterion = LabelSmoothingLoss(
        num_classes=vocab_size,
        padding_idx=pad_idx,
        smoothing=config['label_smoothing']
    )
    base_optimizer = torch.optim.Adam(
        model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
    )
    optimizer = NoamOpt(
        model_size=config['d_model'],
        factor=1,
        warmup=config['warmup_steps'],
        optimizer=base_optimizer
    )
    
    print(f"\n{'='*60}")
    print(f"配置: {config['name']}")
    print(f"参数: d_model={config['d_model']}, num_heads={config['num_heads']}, num_layers={config['num_layers']}, d_ff={config['d_ff']}")
    print(f"参数量: {param_count:,}")
    print(f"训练样本: {config['train_samples']}, 验证样本: {config['val_samples']}")
    print(f"{'='*60}")
    
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_data, criterion, optimizer, device, pad_idx)
        val_loss = validate_epoch(model, val_data, criterion, device, pad_idx)
        
        if epoch % 5 == 0:
            val_acc = calculate_accuracy(model, val_data, device, pad_idx)
        else:
            val_acc = 0.0
        
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch+1:3d}/{config['num_epochs']}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        if val_acc > 0:
            print(f"  Val Acc:    {val_acc:.4f}")
        print(f"  LR:         {optimizer._rate:.6f}")
        print(f"  Time:       {elapsed:.2f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    final_acc = calculate_accuracy(model, val_data, device, pad_idx)
    
    print(f"\n{config['name']} 结果:")
    print(f"  最佳验证 Loss: {best_val_loss:.6f}")
    print(f"  最终准确率:   {final_acc:.4f}")
    
    return {
        'name': config['name'],
        'params': param_count,
        'best_val_loss': best_val_loss,
        'final_acc': final_acc,
        'config': config
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    print("\n" + "="*60)
    print("Attention Is All You Need 论文对比实验")
    print("="*60)
    
    print("\n论文原版实验配置（Table 3）:")
    print("-" * 50)
    print(f"{'配置':<15} {'d_model':>8} {'num_heads':>10} {'num_layers':>10} {'d_ff':>8}")
    print(f"{'Transformer (base)':<15} {512:>8} {8:>10} {6:>10} {2048:>8}")
    print(f"{'Transformer (big)':<15} {1024:>8} {16:>10} {6:>10} {4096:>8}")
    
    print("\n论文原版实验结果（Table 2）:")
    print("-" * 50)
    print(f"{'配置':<15} {'WMT 2014 英德':>18} {'WMT 2014 英法':>18}")
    print(f"{'Transformer (base)':<15} {'27.3 BLEU':>18} {'38.1 BLEU':>18}")
    print(f"{'Transformer (big)':<15} {'28.4 BLEU':>18} {'41.0 BLEU':>18}")
    
    print("\n" + "="*60)
    print("开始对比实验（使用合成翻译数据）")
    print("="*60)
    
    experiments = [
        {
            'name': 'Transformer (base) - 论文配置',
            'd_model': 512,
            'num_heads': 8,
            'd_ff': 2048,
            'num_layers': 6,
            'max_len': 50,
            'dropout': 0.1,
            'batch_size': 32,
            'num_epochs': 30,
            'warmup_steps': 4000,
            'label_smoothing': 0.1,
            'vocab_size': 500,
            'train_samples': 5000,
            'val_samples': 1000
        },
        {
            'name': 'Transformer (big) - 论文配置',
            'd_model': 1024,
            'num_heads': 16,
            'd_ff': 4096,
            'num_layers': 6,
            'max_len': 50,
            'dropout': 0.1,
            'batch_size': 16,
            'num_epochs': 20,
            'warmup_steps': 4000,
            'label_smoothing': 0.1,
            'vocab_size': 500,
            'train_samples': 5000,
            'val_samples': 1000
        },
        {
            'name': '小模型对照 (d_model=128)',
            'd_model': 128,
            'num_heads': 4,
            'd_ff': 512,
            'num_layers': 2,
            'max_len': 50,
            'dropout': 0.1,
            'batch_size': 64,
            'num_epochs': 30,
            'warmup_steps': 1000,
            'label_smoothing': 0.1,
            'vocab_size': 500,
            'train_samples': 5000,
            'val_samples': 1000
        },
        {
            'name': '单层对照 (num_layers=1)',
            'd_model': 256,
            'num_heads': 4,
            'd_ff': 1024,
            'num_layers': 1,
            'max_len': 50,
            'dropout': 0.1,
            'batch_size': 64,
            'num_epochs': 30,
            'warmup_steps': 1000,
            'label_smoothing': 0.1,
            'vocab_size': 500,
            'train_samples': 5000,
            'val_samples': 1000
        },
        {
            'name': '单头对照 (num_heads=1)',
            'd_model': 256,
            'num_heads': 1,
            'd_ff': 1024,
            'num_layers': 2,
            'max_len': 50,
            'dropout': 0.1,
            'batch_size': 64,
            'num_epochs': 30,
            'warmup_steps': 1000,
            'label_smoothing': 0.1,
            'vocab_size': 500,
            'train_samples': 5000,
            'val_samples': 1000
        }
    ]
    
    results = []
    
    for exp in experiments:
        result = run_experiment(exp, device)
        results.append(result)
    
    print("\n" + "="*60)
    print("实验结果汇总")
    print("="*60)
    print(f"{'配置':<30} {'参数量':>12} {'验证Loss':>12} {'准确率':>10}")
    print("-" * 64)
    
    for result in results:
        print(f"{result['name']:<30} {result['params']:>12,} {result['best_val_loss']:>12.6f} {result['final_acc']:>10.4f}")
    
    print("\n" + "="*60)
    print("与论文对比分析")
    print("="*60)
    print("""
1. 参数配置对比：
   - 我们使用论文原版的 d_model、num_heads、num_layers、d_ff
   - 学习率调度使用论文公式：lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
   - 使用论文推荐的 Adam 参数：betas=(0.9, 0.98), eps=1e-9

2. 数据差异：
   - 论文使用 WMT 2014 数据集（约 450 万英德句子对）
   - 我们使用合成数据（5000 个样本），用于快速验证架构正确性

3. 训练条件差异：
   - 论文训练 3.5 天 on 8 P100 GPUs
   - 我们使用单 GPU/CPU，训练少量 epoch

4. 预期结果：
   - Transformer (base) 应比小模型表现更好
   - Transformer (big) 应比 base 表现更好（但需要更多计算资源）
   - 增加层数和头数应提升模型表达能力
   - 单头模型应比多头模型表现差

5. 完整复现论文结果需要：
   - 下载 WMT 2014 数据集（约 45GB）
   - 使用 Byte Pair Encoding (BPE) 分词
   - 8 个 GPU 训练 3.5 天
   - 使用 beam search 解码（论文使用 beam_size=4）
   - 使用 checkpoint averaging
""")


if __name__ == "__main__":
    main()
