import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 28
seq_len = 28
hidden_size = 256
num_layers = 2
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

class RNN(nn.Module):
    def __init__(self, input_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(seq_len * hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out
    
class GRU(nn.Module):
    def __init__(self, input_size, num_layers, num_classes):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # 跟RNN比较 只用改这里
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(seq_len * hidden_size, num_classes)

    def forward(self, x):
        #x.size(0) = batch_size
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # print('shape of h: {}'.format(h0.shape))

        out, _ = self.rnn(x, h0)
        # print('shape of out: {}'.format(out.shape))

        out = out.reshape(out.shape[0], -1)
        # print('shape of out after: {}'.format(out.shape))

        out = self.fc(out)
        return out

train_dataset = datasets.MNIST('data', train=True,
                               transform=transforms.ToTensor(), download=True)

test_dataset = datasets.MNIST('data', train=False,
                              transform=transforms.ToTensor(), download=True)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = GRU(input_size=input_size, num_layers=num_layers, num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for img, label in tqdm(train_dataloader, desc='Training'):
        # print("Data shape before: {} {}".format(img.shape, label.shape))

        img, label = img.to(device).squeeze(1), label.to(device)

        # print("Data shape after: {} {}".format(img.shape, label.shape))

        y = model(img)
        loss = criterion(y, label)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

def check_acc(loader, model):
    if loader.dataset.train:
        print("Check accuracy on training data")
    else:
        print("Check accuracy on test data")
    
    num_corrects, num_samples = 0, 0;
    model.eval()

    with torch.no_grad():
        for x, y in tqdm(loader, desc='Checking'):
            x, y = x.to(device).squeeze(1), y.to(device)

            scores = model(x)
            _, prediction = scores.max(1)
            num_corrects += (prediction == y).sum()
            num_samples += prediction.size(0)
    print('Accuracy is {}'.format(num_corrects / num_samples))

check_acc(train_dataloader, model)
check_acc(test_dataloader, model)

