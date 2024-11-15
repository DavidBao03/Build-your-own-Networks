import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformers, datasets
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
from tqdm import tqdm
from pathlib import Path
import random
import itertools
import numpy as np

MAX_LEN = 64

corpus_movie_path = "../data/cornell movie-dialogs corpus/movie_conversations.txt"
corpus_movie_lines = "../data/cornell movie-dialogs corpus/movie_lines.txt"

with open(corpus_movie_path, "r", encoding="iso-8859-1") as c:
    convs = c.readlines()
with open(corpus_movie_lines, "r", encoding="iso-8859-1") as l:
    lines = l.readlines()

line_dic = {}
for line in lines:
    object = line.split(" +++$+++ ")
    line_dic[object[0]] = object[-1]

pairs = []
for conv in convs:
    ids = eval(conv.split(" +++$+++ ")[-1])

    for i in range(len(ids)):
        qa_pairs = []

        if i == len(ids) - 1:
            break

        first = line_dic[ids[i].strip()]
        second = line_dic[ids[i + 1].strip()]

        qa_pairs.append(" ".join(first.split()[:MAX_LEN]))
        qa_pairs.append(" ".join(second.split()[:MAX_LEN]))
        pairs.append(qa_pairs)

text_data = []
file_count = 0

for pair in tqdm(pairs):
    text_data.append(pair[0])

    if len(text_data) == 10000:
        with open(f'../data/cornell movie-dialogs corpus/text/text_{file_count}.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(text_data))
        text_data = []
        file_count += 1

paths = [str(x) for x in Path('../data/cornell movie-dialogs corpus/text').glob('**/*.txt')]

tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=True
)

tokenizer.train(
    files=paths,
    vocab_size=30000,
    min_frequency=5,
    limit_alphabet=1000,
    wordpieces_prefix='##',
    special_tokens=['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
)

tokenizer.save_model('./bert-it-1', 'bert-it')
tokenizer = BertTokenizer.from_pretrained('./bert-it-1/bert-it-vocab.txt', local_files_only=True)

# ------------------------------------------------------------------------------------------------------------

class BertDataset(Dataset):
    def __init__(self, data_lines, tokenizer, seq_len=64):
        self.data_lines = data_lines
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.data_lines_len = len(self.data_lines)

    def __len__(self):
        return self.data_lines_len
    
    def __getitem__(self, index):
        sentence1, sentence2, is_next_sentence = self.get_sentences(index)

        sentence1, sentence1_label = self.set_random_word(sentence1)
        sentence2, sentence2_label = self.set_random_word(sentence2)

        sentence1 = [self.tokenizer.vocab['[CLS]']] + sentence1 + [self.tokenizer.vocab['[SEP]']]
        sentence2 = sentence2 + [self.tokenizer.vocab['[SEP]']]
        sentence1_label = [self.tokenizer.vocab['[PAD]']] + sentence1_label + [self.tokenizer.vocab['[PAD]']]
        sentence2_label = sentence2_label + [self.tokenizer.vocab['[PAD]']]

        segment_labels = [1 for _ in range(len(sentence1))] + [2 for _ in range(len(sentence2))]

        if len(segment_labels) > self.seq_len:
            segment_labels = segment_labels[:self.seq_len]
        if len(sentence1 + sentence2) > self.seq_len:
            bert_input = (sentence1 + sentence2)[:self.seq_len]
            bert_label = (sentence1_label + sentence2_label)[:self.seq_len]
        else:
            bert_input = sentence1 + sentence2
            bert_label = sentence1_label + sentence2_label

        padding = [self.tokenizer.vocab['[PAD]'] for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding)
        bert_label.extend(padding)
        segment_labels.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_labels,
                  "is_next": is_next_sentence}

        return {key: torch.tensor(value) for key, value in output.items()}


    def set_random_word(self, sentence):
        tokens = sentence.split()
        output_sentence = []
        output_labels = []

        for i, token in enumerate(tokens):
            prob = random.random()
            # remove [cls] and [sep] tokens
            token_id = self.tokenizer(token)["input_ids"][1:-1]

            if prob < 0.15:
                prob /= 0.15

                # replace to [MASK]
                if prob < 0.8:
                    for _ in range(len(token_id)):
                        output_sentence.append(self.tokenizer.vocab['[MASK]'])
                # replace to random word
                elif prob < 0.9:
                    for _ in range(len(token_id)):
                        output_sentence.append(random.randrange(len(self.tokenizer.vocab)))
                else:
                    output_sentence.append(token_id)
                output_labels.append(token_id)
            else:
                output_sentence.append(token_id)
                for _ in range(len(token_id)):
                    output_labels.append(0)

        # avoid list in list
        output_sentence = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output_sentence]))
        output_labels = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output_labels]))
        assert len(output_sentence) == len(output_labels)
        return output_sentence, output_labels
    
    def get_sentences(self, index):
        sentence1, sentence2 = self.get_corpus_line(index)

        if random.random() > 0.5:
            return sentence1, sentence2, 1
        else:
            return sentence1, self.get_random_line(), 0

    def get_corpus_line(self, index):
        return self.data_lines[index][0], self.data_lines[index][1]

    def get_random_line(self):
        return self.data_lines[random.randrange(0, self.data_lines_len)][1]
    

train_data = BertDataset(pairs, seq_len=MAX_LEN, tokenizer=tokenizer)
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

# ------------------------------------------------------------------------------------------------------------

class BertEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, seq_len=64, dropout=0.1):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, embed_size)
        self.position_emb = nn.Embedding(seq_len, embed_size)
        self.segment_emb = nn.Embedding(3, embed_size)
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence, segment_labels):
        position = torch.arange(self.seq_len).unsqueeze(0).to(sequence.device)
        out = self.token_emb(sequence) + self.position_emb(position) + self.segment_emb(segment_labels)
        out = self.dropout(out)
        return out

# Bert Self Attention
class BertSelfAttention(nn.Module):
    def __init__(self, embed_size, heads=8, dropout=0.1):
        super().__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)

        out = self.fc_out(out)
        out = self.dropout(out)
        return out
    
# Bert Feed Forward
class BertFeedForward(nn.Module):
    def __init__(self, embed_size, hidden_size, dropout=0.1):
        super().__init__()

        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
# Bert Block
class BertBlock(nn.Module):
    def __init__(self, embed_size, heads, hidden_size, dropout=0.1):
        super().__init__()

        self.attention = BertSelfAttention(embed_size, heads, dropout)
        self.feed_forward = BertFeedForward(embed_size, hidden_size, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.norm1(attention + query)
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
    
# Bert Model
class Bert(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, seq_len, hidden_size, num_classes, dropout=0.1):
        super().__init__()

        self.embed_size = embed_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.vocab_size = vocab_size

        self.embedding = BertEmbedding(vocab_size, embed_size, seq_len, dropout)
        self.layers = nn.ModuleList(
            [
                BertBlock(embed_size, heads, hidden_size, dropout)
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Linear(embed_size, num_classes)

    def forward(self, x, segment_label):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        x = self.embedding(x, segment_label)

        for layer in self.layers:
            x = layer(x, x, x, mask)

        return x
    
# Next Sentence Prediction
class NextSentencePrediction(nn.Module):
    def __init__(self, embed_size, num_classes=2):
        super().__init__()

        self.fc = nn.Linear(embed_size, num_classes)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.fc(x[:, 0]))
    
# Masked Language Model
class MaskedLanguageModel(nn.Module):
    def __init__(self, embed_size, vocab_size):
        super().__init__()

        self.fc = nn.Linear(embed_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.fc(x))
    
# BertLM Model
class BertLM(nn.Module):
    def __init__(self, bert):
        super().__init__()

        self.bert = bert
        self.next_sentence = NextSentencePrediction(bert.embed_size)
        self.mask_lm = MaskedLanguageModel(bert.embed_size, bert.vocab_size)

    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.next_sentence(x), self.mask_lm(x)
    
    
# ------------------------------------------------------------------------------------------------------------

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")
model = BertLM(Bert(vocab_size=30000, embed_size=512, num_layers=6, heads=8, seq_len=64, hidden_size=2048, num_classes=2))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

# ------------------------------------------------------------------------------------------------------------

def train(model, data_loader, criterion, optimizer, device):
    model.train()
    model.to(device)
    running_loss = 0.0

    for data in tqdm(data_loader):
        data = {key: value.to(device) for key, value in data.items()}

        input_ids = data["bert_input"]
        segment_label = data["segment_label"]
        is_next = data["is_next"]

        optimizer.zero_grad()
        next_sentence, mask_lm = model.forward(input_ids, segment_label)
        loss = criterion(next_sentence, is_next) + criterion(mask_lm.transpose(1, 2), input_ids)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(data_loader)

# ------------------------------------------------------------------------------------------------------------

for epoch in range(5):
    loss = train(model, train_loader, criterion, optimizer, device)
    print(f"Epoch: {epoch + 1}, Loss: {loss}")


    



