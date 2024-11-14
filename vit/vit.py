import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from tqdm import tqdm

class PatchEmbed(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()

        image_size = (image_size, image_size)
        patch_size = (patch_size, patch_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.image_size[0] and W == self.image_size[1], \
               f"Input image size doesn't match"
        
        # (B, C, H, W) -> (B, DIM, H // patch, W // patch)
        # (B, DIM, H // patch, W // patch) -> (B, DIM, (H * W // patch ** 2))
        # (B, DIM, (H * W // patch ** 2)) -> (B,(H * W // patch ** 2), DIM)
        # Since, every patch was embedded.
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x
    
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, dropout_ratio):
        super().__init__()

        self.linear1 = nn.Linear(in_features, hidden_features)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_ratio)
        self.linear2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.dropout(self.gelu(self.linear1(x)))
        x = self.linear2(x)
        return x
    
class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_features, num_heads):
        super().__init__()

        self.c_proj = nn.Linear(hidden_features, 3 * hidden_features)
        self.out_proj = nn.Linear(hidden_features, hidden_features)
        self.dim = hidden_features
        self.num_heads = num_heads

    def forward(self, x):
        B, T, D = x.shape
        assert D // self.num_heads, "must be divied"

        qkv = self.c_proj(x)
        q, k ,v = qkv.split(self.dim, -1)

        q = q.view(B, T, self.num_heads, D // self.num_heads)
        k = k.view(B, T, self.num_heads, D // self.num_heads)
        v = v.view(B, T, self.num_heads, D // self.num_heads)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(2, 1).contiguous().view(B, T, D)
        out = self.out_proj(out)

        return out

    
class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4,
                 drop_ratio=0.,
                 ):
        super().__init__()

        self.ln_1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(hidden_features=dim, num_heads=num_heads)
        self.dropot = nn.Dropout(drop_ratio)

        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(mlp_ratio * dim), dropout_ratio=drop_ratio)

    def forward(self, x):
        x = x + self.dropot(self.attn(self.ln_1(x)))
        x = x + self.dropot(self.mlp(self.ln_2(x)))
        return x
    
class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_c=3, num_classes=1000, 
                 embed_dim = 768, depth=12, num_heads=12, mlp_ratio=4, drop_ratio=0.):
        super().__init__()

        self.num_classes = num_classes

        self.patch_embed = PatchEmbed(image_size, patch_size, in_c, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_toekn = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, drop_ratio) for _ in range(depth)]
        )

        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # (B, C, H, W) -> (B, num_patches, D)
        x = self.patch_embed(x)
        # add cls_token
        cls_token = self.cls_toekn.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        # shape stays
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        # take the classifier token
        x = x[:, 0]
        # (B, 1, D) -> (B, 1, num_classes)
        x = self.head(x)
        return x
    
# Test
# input = torch.tensor(np.ones((10, 3, 224, 224), dtype=np.float32))
# model = VisionTransformer()
# model.eval()
# output = model(input)
# print(output.shape)

training_data = datasets.FashionMNIST(
    root="../data",
    train=True,
    download=False,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="../data",
    train=False,
    download=False,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=False)

train_features, train_labels = next(iter(train_dataloader))
print(train_features.shape)
print(train_labels.shape)

model = VisionTransformer(28, 4, 1, 10, 128, 2, 8)
optimizer = torch.optim.AdamW(model.parameters(), 3e-4)

for epoch in range(10):
    loss_acm = 0.0
    num_correct = 0
    num_total = 0
    model.train()
    for train_features, train_labels in tqdm(train_dataloader, leave=False):
        optimizer.zero_grad()
        y = model(train_features)
        loss = F.cross_entropy(y, train_labels)
        loss.backward()
        optimizer.step()
        loss_acm += loss.detach()
        num_correct += (y.argmax(1) == train_labels).sum()
        num_total += train_labels.shape[0]
    loss_acm /= num_total
    acc = num_correct / num_total
    print(f"epoch: {epoch} | loss: {loss_acm} | acc: {acc}")
    
    model.eval()
    loss_acm = 0.0
    num_correct = 0
    num_total = 0
    for test_features, test_labels in tqdm(test_dataloader, leave=False):
        with torch.no_grad():
            y = model(test_features)
            loss = F.cross_entropy(y, test_labels)
            loss_acm += loss.detach()
            num_correct += (y.argmax(1) == test_labels).sum()
            num_total += test_labels.shape[0]
    loss_acm /= num_total
    acc = num_correct / num_total
    print(f"epoch: {epoch} | loss: {loss_acm} | acc: {acc}")