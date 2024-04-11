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

tokenize_data = True

if tokenize_data:

    # load  data
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

    # save the tokenized data to csv files

input_ids = df_train['input_ids'].tolist()
attention_mask = df_train['attention_mask'].tolist()
labels = df_train['label'].tolist()

val_input_ids = df_dev['input_ids'].tolist()
val_attention_mask = df_dev['attention_mask'].tolist()
val_labels = df_dev['label'].tolist()

def split_into_chunks(tensor, max=10):
    max_length = tensor.size(0)
    chunks = []
    for i in range(0, max_length, 512):
        if (len(chunks) >= max): break
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

val_input_ids = [torch.stack(split_into_chunks(torch.tensor(i))) for i in val_input_ids]
val_attention_mask = [torch.stack(split_into_chunks(torch.tensor(i))) for i in val_attention_mask]
val_input_ids =[torch.squeeze(i, dim=1) for i in val_input_ids]
val_attention_mask =[torch.squeeze(i, dim=1) for i in val_attention_mask]
val_lengths =[i.size(0) for i in val_input_ids]

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

def make_mask(data, lengths, batch_first=True):
    if batch_first:
        max_length = data.size(1)
        batch_size = data.size(0)
    else:
        max_length = data.size(0)
        batch_size = data.size(1)
    mask = torch.zeros((max_length, batch_size), dtype=torch.bool)

    for i, l in enumerate(lengths):
        mask[:l, i] = 1.

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
        #print(bert_output.shape)
        bert_output[torch.isnan(bert_output)] = 0
        sentence_mask = make_mask(input_ids, lengths).to('cuda')

        return self.attention_mlp(bert_output.permute(1,0,2), sentence_mask.T)

dataset = ECHRDataset(input_ids, attention_mask, labels)
val_dataset = ECHRDataset(val_input_ids, val_attention_mask, val_labels)

# create a dataloader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

#bert = BertModel.from_pretrained('nlpaueb/legal-bert-base-uncased')
bert = AutoModel.from_pretrained("distilbert/distilbert-base-uncased") # distilbert-base-uncased instead of bert-base-uncased for memory issues

accelerator = Accelerator(gradient_accumulation_steps=2)

model = Hierbert(bert=bert, hidden_sizes=[768, 16])
optimizer = Adam(model.parameters(), lr=0.001)

loss_function = torch.nn.BCELoss()

model, optimizer, training_dataloader, val_dataloader = accelerator.prepare(
     model, optimizer, dataloader, val_dataloader
)

from tqdm import tqdm
from sklearn.metrics import f1_score

best_val_loss = np.inf
best_model = None

for epoch in range(4):
    model.train()
    train_loss = 0
    for batch in tqdm(training_dataloader, desc=f"Training Epoch {epoch}"):
        optimizer.zero_grad()
        inputs, att, targets, l = batch
        outputs = model(inputs, att, l)
        loss = loss_function(outputs, targets.float())
        accelerator.backward(loss)
        optimizer.step()
        train_loss += loss.item()
        del inputs 
        del att 
        del targets 
        del l
        del outputs 
        del loss
        torch.cuda.empty_cache()

    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f"Validation Epoch {epoch}"):
            inputs, att, targets, l = batch
            outputs = model(inputs, att, l)
            loss = loss_function(outputs, targets.float())
            val_loss += loss.item()
            outputs = torch.round(outputs)
            accuracy = f1_score(targets.cpu().numpy(), outputs.cpu().numpy(), average='weighted')
            val_accuracy += accuracy
            del inputs
            del att
            del targets
            del l
            del outputs
            del loss
            torch.cuda.empty_cache()

    train_loss = train_loss / len(training_dataloader)
    val_loss = val_loss / len(val_dataloader)
    val_accuracy = val_accuracy / len(val_dataloader)
    # update progress bar 
    print(f"Epoch {epoch} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val F1: {val_accuracy}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model.state_dict()

torch.save(best_model, 'hierbert.pt')

# test the model
test_input_ids = df_test['input_ids'].tolist()
test_attention_mask = df_test['attention_mask'].tolist()
test_labels = df_test['label'].tolist()

test_input_ids = [torch.stack(split_into_chunks(torch.tensor(i))) for i in test_input_ids]
test_attention_mask = [torch.stack(split_into_chunks(torch.tensor(i))) for i in test_attention_mask]
test_input_ids =[torch.squeeze(i, dim=1) for i in test_input_ids]
test_attention_mask =[torch.squeeze(i, dim=1) for i in test_attention_mask]
test_lengths =[i.size(0) for i in test_input_ids]

test_dataset = ECHRDataset(test_input_ids, test_attention_mask, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# load the best model
model.load_state_dict(torch.load('hierbert.pt'))

model.eval()
test_loss = 0
test_accuracy = 0
with torch.no_grad():
    for batch in tqdm(test_dataloader, desc=f"Test"):
        inputs, att, targets, l = batch
        outputs = model(inputs, att, l)
        loss = loss_function(outputs, targets.float())
        test_loss += loss.item()
        outputs = torch.round(outputs)
        accuracy = f1_score(targets.cpu().numpy(), outputs.cpu().numpy(), average='weighted')
        test_accuracy += accuracy
        del inputs
        del att
        del targets
        del l
        del outputs
        del loss
        torch.cuda.empty_cache()

test_loss = test_loss / len(test_dataloader)
test_accuracy = test_accuracy / len(test_dataloader)
print(f"Test Loss: {test_loss} - Test F1: {test_accuracy}")

