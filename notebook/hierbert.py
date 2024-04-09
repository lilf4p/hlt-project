import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import transformers

from accelerate import Accelerator
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch

from transformers import BertModel, AutoModel, AutoTokenizer

import sys
sys.path.append('..')
from src.attentionmlp import AttentionMLP

# load tokenized data
path_train ='/storagenfs/l.stoppani/hlt-project/hlt-project/ECHR_Dataset/train.csv'
path_dev ='/storagenfs/l.stoppani/hlt-project/hlt-project/ECHR_Dataset/dev.csv'
path_test ='/storagenfs/l.stoppani/hlt-project/hlt-project/ECHR_Dataset/test.csv'

df_train = pd.read_csv(path_train)
df_dev = pd.read_csv(path_dev)
df_test = pd.read_csv(path_test)

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
def preprocess_function(examples):
    return tokenizer(examples["text"], padding=False)

from datasets import Dataset
df_train = Dataset.from_pandas(df_train)
df_dev = Dataset.from_pandas(df_dev)
df_test = Dataset.from_pandas(df_test)

df_train = df_train.map(preprocess_function, batched= True)
df_dev = df_dev.map(preprocess_function, batched= True)
df_test = df_test.map(preprocess_function, batched= True)

df_train = df_train.with_format(type = "pandas", columns= ['input_ids', 'attention_mask', 'label'] )
df_dev = df_dev.with_format(type = "pandas", columns= ['input_ids', 'attention_mask', 'label'])
df_test = df_test.with_format(type = "pandas", columns= ['input_ids', 'attention_mask', 'label'])

#documents = df_train[['input_ids', 'attention_mask', 'label']]
# convert the series into a list
input_ids = df_train['input_ids'].tolist()
attention_mask = df_train['attention_mask'].tolist()
labels = df_train['label'].tolist()

print(input_ids[0])
print(type(input_ids[0]))

def split_into_chunks(tensor):
    max_length = tensor.size(0)
    chunks = []
    for i in range(0, max_length, 512):
        chunk = tensor[i:i+512]
        chunks.append(chunk)
    pad_size = (0, 512 - len(chunks[-1]))  # Calculate the pad size
    chunks[-1] = torch.nn.functional.pad(chunks[-1], pad_size, 'constant', value=0)  # Pad the chunk with zeros
    return chunks

input_ids = [torch.stack(split_into_chunks(torch.tensor(i))) for i in input_ids]
attention_mask = [torch.stack(split_into_chunks(torch.tensor(i))) for i in attention_mask]
input_ids =[torch.squeeze(i, dim=1) for i in input_ids]
attention_mask =[torch.squeeze(i, dim=1) for i in attention_mask]
lengths =[i.size(0) for i in input_ids]

def collate_fn(data):
    """
       data: is a list of tuples with (input_ids, attention mask, label, length)
    """
    input_ids = [i[0] for i in data]
    attention_mask = [i[1] for i in data]
    labels = [i[2] for i in data]
    lengths = [i[3] for i in data]


    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)
    max_length = torch.max(lengths)
    # pad the input_ids and attention_mask so that they have the same length [max_length, 512]
    for i in range(len(input_ids)):
        pad = torch.zeros((max_length - lengths[i],512), dtype=torch.long)
        input_ids[i] = torch.cat((input_ids[i], pad), dim=0, )
        attention_mask[i] = torch.cat((attention_mask[i], pad), dim=0)

    return torch.stack(input_ids), torch.stack(attention_mask), labels, lengths

class ECHRDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx], self.input_ids[idx].size(0) # last one is the length of the input_ids, used for padding
    # create a dataset

dataset = ECHRDataset(input_ids, attention_mask, labels)
# create a dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
# try to run the model
#bert = BertModel.from_pretrained('nlpaueb/legal-bert-base-uncased')
bert = AutoModel.from_pretrained("distilbert/distilbert-base-uncased")

def make_mask(data, lengths, batch_first=True):
    if batch_first:
        max_length = data.size(1)
        batch_size = data.size(0)
    else:
        max_length = data.size(0)
        batch_size = data.size(1)
    mask = torch.zeros((max_length, batch_size), dtype=torch.bool)

    for i, l in enumerate(lengths):
        mask[i, :l] = 1.

    return mask


class Hierbert(nn.Module):
    def __init__(self, bert, hidden_sizes ):
        super(Hierbert, self).__init__()
        self.bert = bert
        self.attention_mlp = AttentionMLP(768, hidden_sizes)

    def forward(self, input_ids, attention_masks, lengths, bert_require_grad=True):

        max_l = input_ids.size(1)# inputs are already padded
        bert_output = []

        if bert_require_grad:
            self.bert.train()
        else:
            self.bert.eval()

        for i in range(max_l):
            bert_output.append(
                self.bert(input_ids[:,i], attention_masks[:,i]).last_hidden_state[:, 0, :]
            )

        bert_output = torch.stack(bert_output)

        print(bert_output.shape)
        sentence_mask = make_mask(input_ids, lengths).to('cuda')

        return self.attention_mlp(bert_output.permute(1,0,2), sentence_mask.T)
    



device = 'cuda'
model = Hierbert(bert=bert, hidden_sizes=[768,2])

optimizer = Adam(model.parameters(), lr=0.001)
accelerator = Accelerator()

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
loss_function = torch.nn.BCELoss()

model, optimizer, training_dataloader, scheduler = accelerator.prepare(
     model, optimizer, dataloader, scheduler
)

for batch in training_dataloader:
    optimizer.zero_grad()
    inputs, att, targets, l = batch
    inputs = inputs
    targets = targets
    outputs = model(inputs, att, l)
    loss = loss_function(outputs, targets.float())
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()