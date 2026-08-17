"""
Transformer 完整实现 (答案版)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 1. Scaled Dot-Product Attention
# ============================================================
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)

    # scores = Q @ K^T / sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # softmax 归一化
    attn_weights = F.softmax(scores, dim=-1)

    # 加权求和
    output = torch.matmul(attn_weights, V)

    return output, attn_weights


# ============================================================
# 2. Multi-Head Attention
# ============================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        """(batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k)"""
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        """(batch, num_heads, seq_len, d_k) → (batch, seq_len, d_model)"""
        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_Q(Q))
        K = self.split_heads(self.W_K(K))
        V = self.split_heads(self.W_V(V))

        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        output = self.combine_heads(attn_output)
        output = self.W_O(output)

        return output, attn_weights


# ============================================================
# 3. Position-wise Feed-Forward Network
# ============================================================
class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


# ============================================================
# 4. Positional Encoding
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================
# 5. Encoder Layer
# ============================================================
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-Attention + Add & Norm
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # FFN + Add & Norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x


# ============================================================
# 6. Encoder
# ============================================================
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


# ============================================================
# 7. Decoder Layer
# ============================================================
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Masked Self-Attention + Add & Norm
        attn_output, _ = self.masked_self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Cross-Attention + Add & Norm
        cross_output, cross_attn_weights = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_output))

        # FFN + Add & Norm
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_output))

        return x, cross_attn_weights


# ============================================================
# 8. Decoder
# ============================================================
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x, _ = layer(x, enc_output, src_mask, tgt_mask)
        x = self.output_proj(x)
        return x


# ============================================================
# 9. 完整 Transformer
# ============================================================
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 d_ff=2048, num_layers=6, max_len=5000, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encoder(src, src_mask)
        output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        return output


# ============================================================
# 10. Mask 生成工具
# ============================================================
def generate_padding_mask(seq, pad_idx=0):
    """padding mask: (batch, 1, 1, seq_len)"""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def generate_subsequent_mask(seq_len):
    """causal mask: (1, 1, seq_len, seq_len)"""
    return torch.triu(torch.ones(1, 1, seq_len, seq_len), diagonal=1) == 0


# ============================================================
# 11. 优化器 + 学习率调度
# ============================================================
class NoamOpt:
    def __init__(self, model_size, factor=2, warmup=4000, optimizer=None):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5)) * \
               min(step ** (-0.5), step * self.warmup ** (-1.5))


# ============================================================
# 12. Label Smoothing Loss
# ============================================================
class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, padding_idx=0, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        pred = F.log_softmax(pred, dim=-1)

        # 构造平滑后的目标分布
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0  # padding 位置概率置 0

        # 对于 target 为 padding_idx 的样本，将其分布全部置 0
        mask = (target != self.padding_idx).unsqueeze(1)
        true_dist = true_dist * mask

        # KL 散度损失
        loss = -torch.sum(true_dist * pred, dim=-1)
        return loss.sum() / mask.sum()


# ============================================================
# 13. 过拟合测试
# ============================================================
def overfit_test():
    print("========== 过拟合测试 ==========")

    src_vocab_size = 100
    tgt_vocab_size = 100
    d_model = 64
    num_heads = 4
    d_ff = 128
    num_layers = 2
    max_len = 20
    batch_size = 2
    seq_len = 10

    model = Transformer(
        src_vocab_size, tgt_vocab_size,
        d_model=d_model, num_heads=num_heads, d_ff=d_ff,
        num_layers=num_layers, max_len=max_len, dropout=0.0  # 过拟合测试不用 dropout
    )

    # 固定随机数据
    torch.manual_seed(42)
    src = torch.randint(1, src_vocab_size, (batch_size, seq_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, seq_len))
    # 让 target 就是 tgt 右移一位（预测下一个 token），最简单的方式：直接用 tgt 的前 n-1 个作为输入，后 n-1 个作为标签
    tgt_input = tgt[:, :-1]
    tgt_label = tgt[:, 1:]

    src_mask = generate_padding_mask(src, pad_idx=0)
    tgt_mask = generate_subsequent_mask(tgt_input.size(1))

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for step in range(500):
        optimizer.zero_grad()
        output = model(src, tgt_input, src_mask, tgt_mask)
        # output: (batch, seq_len, vocab_size)
        output = output.contiguous().view(-1, tgt_vocab_size)
        loss = criterion(output, tgt_label.contiguous().view(-1))

        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step:3d}  Loss: {loss.item():.6f}")

    print(f"Final Loss: {loss.item():.6f}")
    print("如果 loss 趋近于 0，说明实现基本正确 ✅")


if __name__ == "__main__":
    overfit_test()
