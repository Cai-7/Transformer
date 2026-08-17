"""
Transformer 训练验证脚本 - 复制任务
"""

import torch
import torch.nn as nn
import numpy as np
import os
import time
from transformer import (
    Transformer,
    generate_padding_mask,
    generate_subsequent_mask,
    NoamOpt,
    LabelSmoothingLoss
)


class CopyDataGenerator:
    def __init__(self, vocab_size, seq_len, batch_size, num_samples):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.idx = 0

        self.src_data = []
        self.tgt_data = []
        for _ in range(num_samples):
            length = np.random.randint(5, seq_len + 1)
            src_seq = torch.randint(2, vocab_size, (length,))
            tgt_seq = torch.cat([torch.tensor([1]), src_seq])
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


def train_epoch(model, train_data, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for src, tgt in train_data:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_label = tgt[:, 1:]

        src_mask = generate_padding_mask(src, pad_idx=0).to(device)
        tgt_mask = generate_subsequent_mask(tgt_input.size(1)).to(device)

        optimizer.optimizer.zero_grad()
        output = model(src, tgt_input, src_mask, tgt_mask)

        output = output.contiguous().view(-1, output.size(-1))
        tgt_label = tgt_label.contiguous().view(-1)

        loss = criterion(output, tgt_label)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * (tgt_label != 0).sum().item()
        total_tokens += (tgt_label != 0).sum().item()

    return total_loss / total_tokens


def validate_epoch(model, val_data, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for src, tgt in val_data:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_label = tgt[:, 1:]

            src_mask = generate_padding_mask(src, pad_idx=0).to(device)
            tgt_mask = generate_subsequent_mask(tgt_input.size(1)).to(device)

            output = model(src, tgt_input, src_mask, tgt_mask)

            output = output.contiguous().view(-1, output.size(-1))
            tgt_label = tgt_label.contiguous().view(-1)

            loss = criterion(output, tgt_label)

            total_loss += loss.item() * (tgt_label != 0).sum().item()
            total_tokens += (tgt_label != 0).sum().item()

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


def calculate_accuracy(model, data, device, max_len=20):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for src, tgt in data:
            src = src.to(device)
            tgt = tgt.to(device)

            src_mask = generate_padding_mask(src, pad_idx=0).to(device)
            decode_len = min(tgt.size(1), max_len)
            decoded = greedy_decode(model, src[0:1], src_mask[0:1], decode_len, 1, device)

            target_seq = tgt[0].cpu().numpy()
            target_len = (target_seq != 0).sum()

            decoded_seq = decoded[1:]
            if len(decoded_seq) >= target_len - 1:
                decoded_seq = decoded_seq[:target_len - 1]

            correct += (decoded_seq == target_seq[1:1 + len(decoded_seq)]).sum()
            total += target_len - 1

    return correct / total


def main():
    config = {
        'vocab_size': 10,
        'd_model': 128,
        'num_heads': 8,
        'd_ff': 512,
        'num_layers': 2,
        'max_len': 20,
        'dropout': 0.0,
        'batch_size': 32,
        'num_epochs': 100,
        'warmup_steps': 500,
        'smoothing': 0.0,
        'save_dir': 'checkpoints'
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = Transformer(
        config['vocab_size'],
        config['vocab_size'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        max_len=config['max_len'],
        dropout=config['dropout']
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_data = CopyDataGenerator(
        vocab_size=config['vocab_size'],
        seq_len=config['max_len'],
        batch_size=config['batch_size'],
        num_samples=2000
    )
    val_data = CopyDataGenerator(
        vocab_size=config['vocab_size'],
        seq_len=config['max_len'],
        batch_size=config['batch_size'],
        num_samples=400
    )

    criterion = LabelSmoothingLoss(
        num_classes=config['vocab_size'],
        padding_idx=0,
        smoothing=config['smoothing']
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

    os.makedirs(config['save_dir'], exist_ok=True)

    best_val_loss = float('inf')
    best_acc = 0.0

    print("\n========== 开始训练（复制任务）==========")
    for epoch in range(config['num_epochs']):
        start_time = time.time()

        train_loss = train_epoch(model, train_data, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_data, criterion, device)

        if epoch % 10 == 0:
            val_acc = calculate_accuracy(model, val_data, device, config['max_len'])
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
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'val_loss': val_loss
            }, os.path.join(config['save_dir'], 'best_model.pt'))

        if val_acc > best_acc:
            best_acc = val_acc

    print("\n========== 训练完成 ==========")
    print(f"Best Val Loss: {best_val_loss:.6f}")
    print(f"Best Val Acc:  {best_acc:.4f}")

    print("\n========== 测试解码 ==========")
    checkpoint = torch.load(os.path.join(config['save_dir'], 'best_model.pt'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_seq = torch.tensor([[2, 3, 4, 5, 6]])
    test_mask = generate_padding_mask(test_seq, pad_idx=0)
    print(f"Input:  {test_seq[0].tolist()}")

    decoded = greedy_decode(model, test_seq, test_mask, config['max_len'], 1, device)
    print(f"Output: {decoded.tolist()}")


if __name__ == "__main__":
    main()
