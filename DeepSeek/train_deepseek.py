import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math
import time

@dataclass
class ModelArgs:
    max_batch_size: int = 4
    max_seq_len: int = 512
    vocab_size: int = 50257
    hidden_dim: int = 768
    input_dim: int = 1536
    moe_inter_dim: int = 1408
    n_layers: int = 2
    num_heads: int = 12
    device = 'cpu'
    # moe
    num_experts: int = 6
    num_share_experts: int = 2
    topk: int = 2
    router_scale: float = 1.
    # mla
    # q_lora_rank: int = 0
    # kv_lora_rank: int = 512
    # qk_nope_head_dim: int = 128
    # qk_rope_head_dim: int = 64
    # v_head_dim: int = 128

def precompute_theta_pos_frequencies(head_dim: int, max_seq_len: int, device: str, theta: float = 10000.0):
    # according to the paper, head_dim should be even
    assert head_dim % 2 == 0, "Head dim should be even"
    # Shape: (head_dim / 2), according to the paper, represents theta [0, 2, ..., d - 2]
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape: (head_dim / 2), formula: theta_i = 10000 ^ (-2(i - 1) / dim), for i = 1, 2, ..., d / 2
    theta = 1 / (theta ** (theta_numerator / head_dim)).to(device)
    # Shape: (max_seq_len), according to the paper, represents the position [1, 2, ..., n]
    m = torch.arange(0, max_seq_len, device=device)
    # Shape: (max_seq_len, head_dim / 2)
    freqs = torch.outer(m, theta).float()
    # Shape: (max_seq_len, head_dim / 2)
    # the first numberrepresents the cos, the second numer represents the sin
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor):
    # (b, s, h, d) -> (b, s, h, d / 2, 2) -> (b, s, h, d / 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (b, s, h, d / 2) * (1, s, 1, d / 2) -> (b, s, h, d / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    x_complex = x_complex * freqs_complex
    # (b, s, h, d / 2) -> (b, s, h, d / 2, 2) -> (b, s, h, d)
    x_out = torch.view_as_real(x_complex).reshape(*x.shape)
    return x_out.type_as(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_dim % config.num_heads == 0
        self.c_attn = nn.Linear(config.hidden_dim, config.hidden_dim * 3)
        self.c_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

        self.num_heads = config.num_heads
        self.hidden_dim = config.hidden_dim
    
    def forward(self, x, freq_complex, mask):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.hidden_dim, dim=-1)
        q = q.view(B, T, self.num_heads, C // self.num_heads)
        k = k.view(B, T, self.num_heads, C // self.num_heads)
        v = v.view(B, T, self.num_heads, C // self.num_heads)

        xk = apply_rotary_embeddings(k, freq_complex).to(x.device)
        xv = apply_rotary_embeddings(v, freq_complex).to(x.device)

        q = q.permute(0, 2, 1, 3)
        xk = xk.permute(0, 2, 1, 3)
        xv = xv.permute(0, 2, 1, 3)
        
        attn = F.scaled_dot_product_attention(q, xk, xv, attn_mask=mask)
        attn  = attn.transpose(1, 2).reshape(B, T, C)
        out = self.c_proj(attn)
        return out

class Gate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config.input_dim
        self.topk = config.topk
        self.router_scale = config.router_scale
        self.weight = nn.Parameter(torch.empty(config.num_experts, config.hidden_dim))
        self.bias = nn.Parameter(torch.empty(config.num_experts))
    
    def forward(self, x):
        scores = F.linear(x, self.weight)
        scores = scores.softmax(dim=-1)
        original_scores = scores
        scores = scores + self.bias

        selected_experts = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(dim=-1, index=selected_experts)
        weights *= self.router_scale
        return weights.type_as(x), selected_experts
    
class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, input_dim)
        self.w3 = nn.Linear(input_dim, hidden_dim)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x) * self.w3(x)))

class SharedSparseMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_experts = config.num_experts

        self.topk = config.topk
        self.gate = Gate(config)
        self.experts = nn.ModuleList([Expert(config.hidden_dim, config.moe_inter_dim) for _ in range(config.num_experts)])
        self.shared_experts = Expert(config.hidden_dim, config.moe_inter_dim)
    
    def forward(self, x):
        shape = x.size()
        x = x.view(-1, self.hidden_dim)
        weights, selected_experts = self.gate(x)
        y = torch.zeros_like(x)
        # a tensor records the number of usage of each expert
        counts = torch.bincount(selected_experts.flatten(), minlength=self.num_experts).tolist()
        for i in range(self.num_experts):
            # this expert is not selected by any sample
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top_x = torch.where(selected_experts == i)
            # y[idx] = expert(x[idx]) * weights[idx, top_x, None]
            y.index_add_(0, idx, expert(x[idx]) * weights[idx, top_x, None])
        z = self.shared_experts(x)
        return (y + z).view(shape)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.hidden_dim)
        self.ffn_norm = nn.RMSNorm(config.hidden_dim)
        self.attn = CausalSelfAttention(config)
        self.ffn = SharedSparseMoE(config)
    
    def forward(self, x, freq_complex, mask):
        x = x + self.attn(self.attn_norm(x), freq_complex, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_seq_len = config.max_seq_len
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers = torch.nn.ModuleList()
        for _ in range(config.n_layers):
            self.layers.append(Block(config))
        self.norm = nn.RMSNorm(config.hidden_dim)
        self.output = nn.Linear(config.hidden_dim, config.vocab_size)

        self.freq_complex = precompute_theta_pos_frequencies(config.hidden_dim // config.num_heads, config.max_seq_len * 2, device=config.device)

    def forward(self, tokens, start_pos: int, targets=None):
        seqlen = tokens.size(1)
        h = self.embed(tokens)
        freq_complex = self.freq_complex[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
        for layer in self.layers:
            h = layer(h, freq_complex, mask)
        h = self.norm(h)
        logits = self.output(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open("input.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        self.current_pos = self.B * self.T
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_pos : self.current_pos + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        
        self.current_pos += B * T
        if self.current_pos + (B * T + 1) > len(self.tokens):
            self.current_pos = 0
        return x, y
    
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def main():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    total_batch_size = 524288 # 2 ** 19 ~0.5M
    B = 4
    T = 512

    grad_accum_steps = total_batch_size // (B * T)

    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gadient accumulation steps: {grad_accum_steps}")

    train_loader = DataLoaderLite(B=B, T=T)

    # lower presicion for training
    torch.set_float32_matmul_precision('high')

    config = ModelArgs()
    config['device'] = device
    model = Transformer()
    model.to(device)
    use_compile = False
    if use_compile:
        model = torch.compile(model)

    print(f"model size: {count_parameters(model) // 1048576}M")

    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 10
    max_steps = 50
    def get_lr(it):
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        
        if it > max_steps:
            return min_lr
        
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)

    for step in range(max_steps):
        t0 = time.time()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, 0, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step()
        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
        tokens_per_sec = tokens_processed / (t1 - t0)

        print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.4f}ms | tok/sec = {tokens_per_sec:.2f}")

if __name__ == "__main__":
    main()