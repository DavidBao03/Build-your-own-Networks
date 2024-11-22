import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from einops.layers.torch import Rearrange
from tqdm import tqdm

from vit import PatchEmbed, Block

from matplotlib import pyplot as plt

# random masking
def random_masking(x, mask_ratio=0.75):
    B, T, D = x.shape
    id_len = int(T * (1 - mask_ratio))

    nosie = torch.randn(B, T, device=x.device)
    id_shuffle = torch.argsort(nosie, dim=1)
    id_restore = torch.argsort(id_shuffle, dim=1)

    id_keep = id_restore[:, :id_len]
    x = torch.gather(x, dim=1, index=id_keep.unsqueeze(-1).repeat(1, 1, D))

    mask = torch.ones(B, T, device=x.device)
    mask[:, :id_len] = 0
    mask = torch.gather(mask, dim=1, index=id_restore)

    return x, mask, id_restore

# MAE Auto encoder
class MAE(nn.Module):
    def __init__(self, embed_dim=1024, decoder_embed_dim=512, patch_size=16, num_head=16, encoder_num_layers=24, decoder_num_layers=8, in_channels=3, img_size=224):
        super(MAE, self).__init__()

        self.patch_embed = PatchEmbed(image_size=img_size, patch_size=patch_size, in_c=in_channels, embed_dim=embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size//patch_size)**2, decoder_embed_dim), requires_grad=False)
        self.decoder_pred = nn.Linear(decoder_embed_dim, in_channels * patch_size * patch_size)

        self.project = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=patch_size**2 * in_channels, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)

        self.encoder_transformer = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_head
            )
            for _ in range(encoder_num_layers)
        ])
        self.decoder_transformer = nn.ModuleList([
            Block(
                dim=decoder_embed_dim, num_heads=num_head
            )
            for _ in range(decoder_num_layers)
        ])

    def encoder(self, x):
        # print("X shape: ", x.shape)

        x = self.patch_embed(x)

        # print("X shape after patch embed: ", x.shape)

        x, mask, id_restore = random_masking(x)

        # print("X shape after random masking: ", x.shape)

        for blk in self.encoder_transformer:
            x = blk(x)

        # print("X shape after encoder transformer: ", x.shape)

        return x, mask, id_restore
    
    def decoder(self, x, id_restore):
        
        x = self.decoder_embed(x)

        # print("X shape after decoder embed: ", x.shape)
        # print("ID restore shape: ", id_restore.shape)

        mask_tokens = self.mask_token.repeat(x.shape[0], id_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat((x, mask_tokens), dim=1)
        x = torch.gather(x_, dim=1, index=id_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2]))

        # print("X shape after decoder masking: ", x.shape)

        x = x + self.pos_embed

        for blk in self.decoder_transformer:
            x = blk(x)

        x = self.decoder_pred(x)

        # print("X shape after decoder transformer: ", x.shape)

        return x
    
    def loss(self, x, img, mask):
        x_hat = self.project(img)

        # print("X hat shape: ", x_hat.shape)
        # print("Mask shape: ", mask.shape)

        loss = (x - x_hat) ** 2
        loss = loss.mean(dim=-1) 
    
        loss = (loss * mask).sum() / mask.sum()
        return loss
    
    def forward(self, img):
        x, mask, id_restore = self.encoder(img)
        pred = self.decoder(x, id_restore)
        loss = self.loss(pred, img, mask)
        return loss, pred, mask
    
# test main
if __name__ == '__main__':
    # test
    # img = torch.randn(4, 1, 28, 28)
    # model = MAE(embed_dim=128, decoder_embed_dim=64, patch_size=4, num_head=8, encoder_num_layers=2, decoder_num_layers=1, in_channels=1, img_size=28)
    # loss, pred, mask = model(img)
    # print(loss)
    # print(pred.shape)
    # print(mask.shape)

    transform = transforms.Compose([transforms.ToTensor()])
    training_data = datasets.FashionMNIST(
        root="../data", train=True, download=False, transform=transform
    )
    test_data = datasets.FashionMNIST(
        root="../data", train=False, download=False, transform=transform
    )

    train_dataloader = DataLoader(training_data, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=4, shuffle=False)

    sample_images, train_labels = next(iter(train_dataloader))
    print(sample_images.shape)
    print(train_labels.shape)

    model = MAE(embed_dim=128, decoder_embed_dim=64, patch_size=4, num_head=8, encoder_num_layers=2, decoder_num_layers=1, in_channels=1, img_size=28)
    model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), 3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # train
    for epoch in range(10):
        loss_acm = 0.0
        num_total = 0
        model.train()
        for train_features, train_labels in tqdm(train_dataloader, leave=False):
            optimizer.zero_grad()
            loss, pred, mask = model(train_features)
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_acm += loss.detach() * train_labels.shape[0]
            num_total += train_labels.shape[0]
        loss_acm /= num_total
        print(f"epoch: {epoch} | loss: {loss_acm}")
        
        model.eval()
        loss_acm = 0.0
        num_total = 0
        for test_features, test_labels in tqdm(test_dataloader, leave=False):
            with torch.no_grad():
                loss, pred, mask = model(test_features)

                loss_acm += loss.detach() * test_labels.shape[0]
                num_total += test_labels.shape[0]
        loss_acm /= num_total
        print(f"epoch: {epoch} | loss: {loss_acm}")

        # Visualize
        _, pred, _ = model(sample_images)
        fig, axs = plt.subplots(2, 4)
        for i in range(4):
            axs[0, i].imshow(sample_images[i].squeeze().numpy(), cmap='gray')
            axs[1, i].imshow(pred[i].detach().cpu().numpy().reshape(28, 28), cmap='gray')
        
        plt.show()
