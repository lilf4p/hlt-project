import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import pandas as pd
import torch.utils

tokenizer = BertTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
class BertAttentionClassifier(nn.Module):
    def __init__(self, num_chunks, max_length, bert_model_name='nlpaueb/legal-bert-base-uncased'):
        super(BertAttentionClassifier, self).__init__()
        self.num_chunks = num_chunks
        self.max_length = max_length
        
        # Load pre-trained BERT model and tokenizer
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Attention layer
        self.attention = nn.MultiheadAttention(embed_dim=self.bert.config.hidden_size, num_heads=1)
        
        # Linear layer for classification
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        # Tokenize and encode each text chunk using BERT tokenizer
        
        
        # Extract BERT outputs (last hidden states)
        bert_outputs = [self.bert( ids, mask).pooler_output
                        for (ids, mask)  in zip(input_ids, attention_mask)]
    
        # Stack BERT outputs along the sequence dimension
        stacked_outputs = torch.stack(bert_outputs, dim=1)  # shape: (batch_size, num_chunks, hidden_size)
        
        # Apply attention across all BERT outputs
        attention_output, _ = self.attention(stacked_outputs.transpose(0, 1),  # (num_chunks, batch_size, hidden_size)
                                             stacked_outputs.transpose(0, 1),  # (num_chunks, batch_size, hidden_size)
                                             stacked_outputs.transpose(0, 1))  # (num_chunks, batch_size, hidden_size)
        
        # Average pooling over the sequence dimension (num_chunks)
        pooled_output = attention_output.mean(dim=0)  # (batch_size, max_length, hidden_size)
        
        # Apply linear layer for classification
        logits = self.fc(pooled_output)  # (batch_size, max_length, 1)
        
        # Squeeze logits to remove extra dimension
        logits = logits.squeeze(dim=-1)  # (batch_size, max_length)
        
        return logits



# load tokenized data
path_dev ='../ECHR_Dataset_Tokenized/legal-bert-base-uncased/df_dev_tokenized.pkl'
path_train ='../ECHR_Dataset_Tokenized/legal-bert-base-uncased/df_train_tokenized.pkl'
path_test ='../ECHR_Dataset_Tokenized/legal-bert-base-uncased/df_test_tokenized.pkl'

df_train = pd.read_pickle(path_train)
df_dev = pd.read_pickle(path_dev)
df_test = pd.read_pickle(path_test)
documents = df_train[['input_ids', 'attention_mask', 'label']]
documents_dev = df_dev[['input_ids', 'attention_mask', 'label']]
documents_test = df_test[['input_ids', 'attention_mask', 'label']]

# convert the series into a list
input_ids = documents.input_ids.tolist()
attention_mask = documents.attention_mask.tolist()
labels = documents.label.tolist()

input_ids_dev = documents_dev.input_ids.tolist()
attention_mask_dev = documents_dev.attention_mask.tolist()
labels_dev = documents_dev.label.tolist()

input_ids_test = documents_test.input_ids.tolist()
attention_mask_test = documents_test.attention_mask.tolist()
labels_test = documents_test.label.tolist()

input_ids = [torch.stack(i) for i in input_ids]
attention_mask = [torch.stack(i) for i in attention_mask]

input_ids_dev = [torch.stack(i) for i in input_ids_dev]
attention_mask_dev = [torch.stack(i) for i in attention_mask_dev]

input_ids_test = [torch.stack(i) for i in input_ids_test]
attention_mask_test = [torch.stack(i) for i in attention_mask_test]

input_ids =[torch.squeeze(i, dim=1) for i in input_ids]
attention_mask =[torch.squeeze(i, dim=1) for i in attention_mask]

input_ids_dev =[torch.squeeze(i, dim=1) for i in input_ids_dev]
attention_mask_dev =[torch.squeeze(i, dim=1) for i in attention_mask_dev]

input_ids_test =[torch.squeeze(i, dim=1) for i in input_ids_test]
attention_mask_test =[torch.squeeze(i, dim=1) for i in attention_mask_test]
lengths =[i.size(0) for i in input_ids]

lengths_dev =[i.size(0) for i in input_ids_dev]

lengths_test =[i.size(0) for i in input_ids_test]
def collate_fn(data, max_chunks=3):
   """
   data: is a list of tuples with (input_ids, attention mask, label, length)
   """
   input_ids = [i[0] for i in data][:max_chunks]
   attention_mask = [i[1] for i in data][:max_chunks]
   labels = [i[2] for i in data]
   lengths = [i[3] for i in data]


   labels = torch.tensor(labels)
   lengths = torch.tensor(lengths)
   max_length = torch.max(lengths)
   max_length = torch.min(max_length, max_chunks * torch.ones_like(max_length))
   print(max_length)
   # truncate input_ids and attention_mask to max_length
   input_ids = [i[:max_length] for i in input_ids]
   attention_mask = [i[:max_length] for i in attention_mask]
   # pad input_ids and attention_mask to max_length
   input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
   attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
   
   return input_ids, attention_mask, labels, lengths


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
eval_dataset = ECHRDataset(input_ids_dev, attention_mask_dev, labels_dev)
test_dataset = ECHRDataset(input_ids_test, attention_mask_test, labels_test)
# number of samples
print(len(dataset))
# create a dataloader
collate_fn_10 = lambda x: collate_fn(x, 10)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=collate_fn_10)
eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=10, shuffle=True, collate_fn=collate_fn_10)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True, collate_fn=collate_fn_10)
# make a subset of the dataset


subset = torch.utils.data.Subset(dataset, range(100))
dataloader = torch.utils.data.DataLoader(subset, batch_size=8, shuffle=True, collate_fn=collate_fn_10)
eval_subset = torch.utils.data.Subset(eval_dataset, range(100))
eval_dataloader = torch.utils.data.DataLoader(eval_subset, batch_size=4, shuffle=True, collate_fn=collate_fn)