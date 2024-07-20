import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets
from torchvision import transforms
from pytorch_resnet import ResNet50
from tqdm import tqdm
import numpy as np

training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=transforms.Compose([ 
        transforms.Resize(28),
        transforms.ToTensor()
    ])
)

test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=transforms.Compose([ 
        transforms.Resize(28),
        transforms.ToTensor()
    ])
)

train_set, valid_set = random_split(training_data, [0.8, 0.2])

train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
valid_dataloader = DataLoader(valid_set, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

model = ResNet50(image_channels=1, num_classes=10)
loss_fn = nn.CrossEntropyLoss()
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

def train(model, epoch, train_dataloader, valid_dataloader, loss_fn, optimizer, device):
    for e in range(epoch):
        model.train()
        for img, label in tqdm(train_dataloader, desc="train epoch{}".format(e + 1)):
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()
        
        vl = 0
        model.eval()
        for img, label in tqdm(valid_dataloader, desc="valid epoch{}".format(e + 1)):
            img, label = img.to(device), label.to(device)
            out = model(img)
            vl += loss_fn(out, label).item()
        vl = vl / len(valid_dataloader.dataset)
        print("epoch {}: loss: {}".format(e + 1, vl))

def test(model, test_dataloader, loss_fn, device):
    model.eval()
    test_loss = 0
    gt_labels = []
    pred_labels = []
    with torch.no_grad():
        for data, label in tqdm(test_dataloader, desc="test"):
            data, label = data.to(device), label.to(device)
            output = model(data)
            preds = torch.argmax(output, 1)
            gt_labels.append(label.cpu().data.numpy())
            pred_labels.append(preds.cpu().data.numpy())
            loss = loss_fn(output, label)
            test_loss += loss.item()*data.size(0)
    test_loss = test_loss/len(test_dataloader.dataset)
    gt_labels, pred_labels = np.concatenate(gt_labels), np.concatenate(pred_labels)
    acc = np.sum(gt_labels==pred_labels)/len(pred_labels)
    print('Test Loss: {:.6f}, Accuracy: {:6f}'.format(test_loss, acc))

train(model, 1, train_dataloader, valid_dataloader, loss_fn=loss_fn, optimizer=optimizer, device=device)
test(model, test_dataloader, loss_fn, device=device)
