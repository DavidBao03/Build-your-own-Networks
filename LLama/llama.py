import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    input_dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32                   # number of heads in query only
    n_kv_heads: Optional[int] = None    # number of heads in shared key and value
    vocab_size: Optional[int] = -1      # specify when the tokenizer is ready

    # feedforward arguments
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None

    norm_eps: float = 1e-5

    # KV cache arguments
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None

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

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # (b, s, h, d) -> (b, s, h, d / 2, 2) -> (b, s, h, d / 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (b, s, h, d / 2) * (1, s, 1, d / 2) -> (b, s, h, d / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    x_complex = x_complex * freqs_complex
    # (b, s, h, d / 2) -> (b, s, h, d / 2, 2) -> (b, s, h, d)
    x_out = torch.view_as_real(x_complex).reshape(*x.shape)
    return x_out.type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n_rep: int):
    batch_size, seq_len, h_kv, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        # (b, s, h_kv, head_dim) -> 
        # (b, s, h_kv, 1, head_dim) ->
        # (b, s, h_kv, n_rep, head_dim) ->
        # (b, s, h_kv * n_rep, head_dim)
        x[:, :, :, None, :]
        .expand(batch_size, seq_len, h_kv, n_rep, head_dim)
        .reshape(batch_size, seq_len, h_kv * n_rep, head_dim)
    )

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # The gamma parameters
        # (dim)
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (b, s, d) -> (b, s, d)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        # (d) * (b, s, d) -> (b, s, d)
        return self.weight * self._norm(x)

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.input_dim
        self.n_heads_kv = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        self.n_heads_q = args.n_heads
        self.head_dim = args.input_dim // args.n_heads
        # The repeated number of kv heads
        self.n_rep = args.n_heads // self.n_heads_kv

        self.wk = nn.Linear(self.dim, self.head_dim * self.n_heads_kv, bias=False)
        self.wv = nn.Linear(self.dim, self.head_dim * self.n_heads_kv, bias=False)
        self.wq = nn.Linear(self.dim, self.head_dim * self.n_heads_q, bias=False)
        self.wo = nn.Linear(self.head_dim * self.n_heads_q, self.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_heads_kv, self.head_dim), device=args.device)
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_heads_kv, self.head_dim), device=args.device)

        self.device = args.device

    def forward(self, x: torch.Tensor, start_pos: int, freq_complex: torch.Tensor):
        # (b, s, d)
        batch_size, seq_len, _ = x.shape

        # (b, s, d) -> (b, s, h_q * head_dim)
        xq = self.wq(x)
        # (b, s, d) -> (b, s, h_kv * head_dim)
        xk = self.wk(x)
        xv = self.wv(x)

        # (b, s, h_kv * head_dim) -> (b, s, h_kv, head_dim)
        xk = xk.view(batch_size, seq_len, self.n_heads_kv, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_heads_kv, self.head_dim)
        # (b, s, h_q * head_dim) -> (b, s, h_q, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)

        # apply rotary embeddings
        # Does not change the shape
        xk = apply_rotary_embeddings(xk, freq_complex, self.device)
        xv = apply_rotary_embeddings(xv, freq_complex, self.device)

        # add KV into cache
        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        # retrieve KV from cache
        keys = self.cache_k[:batch_size, :start_pos + seq_len]
        values = self.cache_v[:batch_size, :start_pos + seq_len]

        # expand KV for Attention
        # (b, s, h_kv, head_dim) -> (b, s, h_kv * n_rep, head_dim)
        # excatly the same as (b, s, h_q, head_dim)
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # (b, s, h_q, head_dim) -> (b, h_q, s, head_dim)
        xq = xq.transpose(1, 2)
        # (b, s_kv, h_kv * n_rep, head_dim) -> (b, h_kv * n_rep, s_kv, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # (b, h_q, s, head_dim) @ (b, h_kv * n_rep, head_dim, s_kv) -> (b, h_q, s, s_kv)
        score = torch.matmul(xq, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        score = F.softmax(score, dim=-1).type_as(xq)
 
        # (b, h_q, s, s_kv) @ (b, h_kv * n_rep, s_kv, head_dim) -> (b, h_q, s, head_dim)
        out = torch.matmul(score, values)
        # (b, h_q, s, head_dim) -> (b, s, h_q * head_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        # (b, s, h_q * head_dim) -> (b, s, d)
        out = self.wo(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = 4 * args.input_dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(hidden_dim * args.ffn_dim_multiplier)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.input_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.input_dim, bias=False)
        self.w3 = nn.Linear(args.input_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        swish = nn.SiLU(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)
        return x
        

class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.input_dim
        self.n_heads = args.n_heads
        self.head_dim = self.dim // self.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # Normalization BEFORE the attention
        self.attention_norm = RMSNorm(self.dim, eps=args.norm_eps)
        # Normalization BEFORE the feed forward
        self.ffn_norm = RMSNorm(self.dim, eps=args.norm_eps)
    
    def forward(self, x: torch.Tensor, start_pos: int, freq_complex: torch.Tensor):
        # (b, s, d) + (b, s, d) -> (b, s, d)
        h = x + self.attention(self.attention_norm(x), start_pos, freq_complex)
        # (b, s, d) + (b, s, d) -> (b, s, d)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1, "Vocab size should be specified when the tokenizer is ready"

        self.args = args
        self.vocab_size = args.vocab_size

        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.input_dim)

        self.layers = nn.ModuleList([EncoderBlock(args) for _ in range(args.n_layers)])

        self.norm = RMSNorm(args.input_dim, eps=args.norm_eps)

        self.output = nn.Linear(args.input_dim, self.vocab_size, bias=False)

        self.freq_complex = precompute_theta_pos_frequencies(self.args.input_dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)

    def forward(self, x: torch.Tensor, start_pos: int):
        bacth_size, seq_len = x.shape

        assert seq_len == 1, "Expected sequence length to be 1"

        # (b, s) -> (b, s, d)
        h = self.tok_embeddings(x)

        # retrive the frequency for the current position
        freq_complex = self.freq_complex[start_pos : start_pos + seq_len]

        # (b, s, d) -> (b, s, d)
        for layer in self.layers:
            h = layer(h, start_pos, freq_complex)
        h = self.norm(h)

        # (b, s, d) -> (b, s, vocab_size)
        output = self.output(h).float()
        return output
        
