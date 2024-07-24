import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np
import spacy
import random
from tqdm import tqdm
from utils import save_checkpoint, load_checkpoint, bleu, translate_sentence

print('loading eng...')
spacy_en = spacy.load('en_core_web_sm')
print('loading ger...')
spacy_ger = spacy.load('de_core_news_sm')
print('loading successful')

def tokenizer_eng(text):
    return [data.text for data in spacy_en.tokenizer(text)]

def tokenizer_ger(text):
    return [data.text for data in spacy_ger.tokenizer(text)]

german = Field(tokenize=tokenizer_ger, lower=True,
               init_token='<sos>', eos_token='<eos>')

english = Field(tokenize=tokenizer_eng, lower=True,
               init_token='<sos>', eos_token='<eos>')

train_data, valid_data, test_data = Multi30k.splits(root='data', exts=('.de', '.en'), fields=(german, english))

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                           bidirectional=True)
        
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        # x shape: [seq_len, batch_size, input_size]
        embedding = self.dropout(self.embedding(x))

        #embedding shape: [seq_len, batch_size, embedding_size]
        hidden_states, (hidden, cell) = self.rnn(embedding)

        hidden = torch.cat((hidden[0:1], hidden[1:2]), dim=2)
        cell = torch.cat((cell[0:1], cell[1:2]), dim=2)

        hidden = self.fc_hidden(hidden)
        cell = self.fc_cell(cell)

        return hidden_states, hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, output_size, embedding_size, hidden_size, num_layers, p):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(input_size=hidden_size * 2 + embedding_size, hidden_size=hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

        self.energy = nn.Linear(hidden_size * 3, 1)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
    
    def forward(self, x, encoder_states, hidden, cell):
        # x shape: [batch_size, input_size] -> [1, batch_size, input_size]
        x = x.unsqueeze(0)
 
        # x shape: [1, batch_size, input_size]
        embedding = self.dropout(self.embedding(x))

        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)
        # h_reshaped: (seq_length, N, hidden_size*2)

        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))
        # energy: (seq_length, N, 1)

        attention = self.softmax(energy)
        # attention: (seq_length, N, 1)

        # attention: (seq_length, N, 1), snk
        # encoder_states: (seq_length, N, hidden_size*2), snl
        # we want context_vector: (1, N, hidden_size*2), i.e knl
        context_vector = torch.einsum("snk,snl->knl", attention, encoder_states)

        rnn_input = torch.cat((context_vector, embedding), dim=2)
        # rnn_input: (1, N, hidden_size*2 + embedding_size)

        #embedding shape: [1, batch_size, embedding_size]
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        #output shape: [1, batch_size, hidden_size]
        pred = self.fc(output)

        #pred shape: [1, batch_size, output_size] -> [batch_size, output_size]
        pred = pred.squeeze(0)

        return pred, hidden, cell

class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size)

        encoder_states, hidden, cell = self.encoder(source)

        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)

            outputs[t] = output

            best_guess = output.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess
        
        return outputs
    
load_model = False
    
num_epochs = 1
learing_rate = 3e-4
batch_size = 32

input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
embedding_size_encoder = 256
embedding_szie_decoder = 256
hidden_size = 1024
num_layers = 1
encoder_dropout = 0.5
decoder_dropout = 0.5

src_sentence = "ein mann in einem blauen hemd steht auf einer leiter und putzt ein fenster."

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iter, valid_iter, test_iter = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_within_batch = True,
    sort_key = lambda x: len(x.src),
    device = device
)

encoder_net = Encoder(input_size_encoder, embedding_size_encoder, 
                      hidden_size, num_layers, encoder_dropout)

decoder_net = Decoder(input_size_decoder, output_size, embedding_szie_decoder,
                      hidden_size, num_layers, decoder_dropout)

model = Seq2seq(encoder_net, decoder_net).to(device)

pad_idx = english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(), lr=learing_rate)
best_loss = 1e3

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

def train(model, train_iter, valid_iter, optimizer, criterion, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_iter, desc='[Training Epoch {} / {}]'.format(epoch, num_epochs)):
            source = batch.src.to(device)
            target = batch.trg.to(device)

            output = model(source, target).to(device)
            output = output[1:].reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            optimizer.step()
        
        model.eval()
        translation = translate_sentence(model, src_sentence, german, english, device)
        print(" ".join(translation))

        total_loss = 0
        for batch in tqdm(valid_iter, desc='[Validing]'):
            source = batch.src.to(device)
            target = batch.trg.to(device)

            with torch.no_grad():
                output = model(source, target).to(device)
                output = output[1:].reshape(-1, output.shape[2])
                target = target[1:].reshape(-1)

                loss = criterion(output, target)
                total_loss += loss

        print('[Epoch {} / {}, loss {}]'.format(epoch, num_epochs, total_loss / len(valid_data)))

        if total_loss < best_loss:
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            save_checkpoint(checkpoint)
            best_loss = total_loss

train(model, train_iter, valid_iter, optimizer, criterion, num_epochs, device)
score = bleu(test_data[1:101], model, german, english, device)
print('BLEU score is {}'.format(score * 100))




