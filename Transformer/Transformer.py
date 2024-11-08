import torch
import torch.nn as nn

class MutiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, heads):
        super(MutiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads

        assert (
            self.head_dim * heads == d_model
        ), "Embedding size needs to be divisible by heads"

        self.value_fc = nn.Linear(self.d_model, self.d_model)
        self.query_fc = nn.Linear(self.d_model, self.d_model)
        self.key_fc = nn.Linear(self.d_model, self.d_model)
        self.fc_out = nn.Linear(self.d_model, self.d_model)

    def forward(self, value, key, query, mask):
        N = value.shape[0]
        val_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        value = self.value_fc(value)
        key = self.key_fc(key)
        query = self.query_fc(query)

        # [N, seq_len, d_model] -> [N, seq_len, heads, head_dim]
        value = value.reshape(N, val_len, self.heads, self.head_dim)
        key = key.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        # query shape: [N, query_len, heads, head_dim]
        # key shape: [N, key_len, heads, head_dim]
        # energy shape: [N, heads, query_len, key_len]
        energy = torch.einsum("nqhd,nkhd->nhqk", [query, key])

        if mask is not None:
            energy = energy.masked_fill(mask==0, float("-1e20"))


        energy = torch.softmax(energy / (self.d_model ** (1/2)), dim=3)

        # energy shape: [N, heads, query_len, key_len]
        # value shape: [N, val_len, heads, head_dim]
        # attention shape: [N, query_len, heads, head_dim]
        attention = torch.einsum("nhql,nlhd->nqhd", [energy, value])

        # [N, seq_len, heads, head_dim] -> [N, seq_len, d_model]
        attention = attention.reshape(N, query_len, -1)

        attention = self.fc_out(attention)

        return attention

class TransformerBlock(nn.Module):
    def __init__(self, d_model, heads, expansion, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MutiHeadSelfAttention(d_model, heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.ReLU(),
            nn.Linear(d_model * expansion, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        x = self.attention(value, key, query, mask)
        out1 = self.dropout(self.norm1(x + query))

        y = self.mlp(out1)
        out2 = self.dropout(self.norm2(y + out1))

        return out2
    
class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        max_length,
        heads,
        expansion,
        dropout,
        num_layers,
        device
    ):
        super(Encoder, self).__init__()
        self.device = device
        self.d_model = d_model
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    heads,
                    expansion=expansion,
                    dropout=dropout
                )
                for _ in range(num_layers)
            ] 
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)
        
        return out
    
class DecoderBlock(nn.Module):
    def __init__(self, d_model, heads, expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.device = device

        self.attention = MutiHeadSelfAttention(d_model, heads)
        self.transformer_block = TransformerBlock(d_model, heads, expansion, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, key, value, trg_mask, src_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(x + attention))
        out = self.transformer_block(value, key, query, src_mask)

        return out
    
class Decoder(nn.Module):
    def __init__(self, d_model, heads, expansion, dropout, num_layers, trg_vocab_size, max_length, device):
        super(Decoder,self).__init__()
        self.device = device
        self.d_model = d_model
        self.word_embedding = nn.Embedding(trg_vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    d_model,
                    heads,
                    expansion=expansion,
                    dropout=dropout,
                    device=device
                ) for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(d_model, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key, value, trg_mask, src_mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, key, value, trg_mask, src_mask)
        
        out = self.fc_out(out)

        return out

    
class Transformer(nn.Module):
    def __init__(self, 
                 src_vocab_size, 
                 trg_vocab_size, 
                 src_pad_idx, 
                 trg_pad_idx,
                 d_model=512,  
                 heads=8, 
                 expansion=4, 
                 dropout=0.1, 
                 device='cpu', 
                 max_length=100, 
                 num_layers=6):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, max_length, heads, expansion, dropout, num_layers, device)
        self.decoder = Decoder(d_model, heads, expansion, dropout, num_layers, trg_vocab_size, max_length, device)

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
    
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )

        return trg_mask.to(self.device) 
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_out = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_out, enc_out, trg_mask, src_mask)

        return out
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], 
                      [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], 
                        [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(
        device
    )
    print(x.shape)
    print(trg.shape)
    out = model(x, trg[:, :-1])
    print(out.shape)
