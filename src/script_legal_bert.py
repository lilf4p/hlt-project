import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import pandas as pd
from tqdm import tqdm

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
    input_ids = [i[0] for i in data]
    attention_mask = [i[1] for i in data]
    labels = [i[2] for i in data]
    lengths = [i[3] for i in data]


    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)
    max_length = torch.max(lengths)
    max_length = torch.min(max_length, max_chunks * torch.ones_like(max_length))
    # truncate input_ids and attention_mask to max_length
    input_ids = [i[:max_length] for i in input_ids]
    attention_mask = [i[:max_length] for i in attention_mask]
    lengths = torch.min(lengths, max_chunks*torch.ones_like(lengths))
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
eval_dataset = ECHRDataset(input_ids_dev, attention_mask_dev, labels_dev)
test_dataset = ECHRDataset(input_ids_test, attention_mask_test, labels_test)
# number of samples
print(len(dataset))
# make a subset of the dataset

collate_fn_10 = lambda x: collate_fn(x, 4)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=True, collate_fn=collate_fn_10)
eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn_10)
x=next(iter(eval_dataloader))
print(x[0].shape, x[1].shape, x[2].shape, x[3].shape)
model_name = 'prajjwal1/bert-small'
tokenizer = BertTokenizer.from_pretrained(model_name)
class BertAttentionClassifier(nn.Module):
    def __init__(self, num_chunks, max_length, bert_model_name='nlpaueb/legal-bert-base-uncased'):
        super(BertAttentionClassifier, self).__init__()
        self.num_chunks = num_chunks
        self.max_length = max_length
        
        # Load pre-trained BERT model and tokenizer
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Attention layer
        self.attention = nn.MultiheadAttention(embed_dim=self.bert.config.hidden_size, num_heads=1)
        self.relu=nn.ReLU()
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
        attention_output, _ = self.attention(stacked_outputs,  # (num_chunks, batch_size, hidden_size)
                                             stacked_outputs,  # (num_chunks, batch_size, hidden_size)
                                             stacked_outputs)  # (num_chunks, batch_size, hidden_size)
        attention_output = self.relu(attention_output)
        # Average pooling over the sequence dimension (num_chunks)
        pooled_output = attention_output.mean(dim=0)  # (batch_size, max_length, hidden_size)

        # Apply linear layer for classification
        logits = self.fc(pooled_output)  # (batch_size, 1)
        
        # Squeeze logits to remove extra dimension
        logits = logits.squeeze(dim=-1)  # (batch_size, max_length)
        
        return logits



# train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BertAttentionClassifier(num_chunks=3, max_length=10, bert_model_name='nlpaueb/legal-bert-base-uncased')
model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-6)
num_epochs=2
best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_samples = 0
    correct_samples = 0
    train_bar = tqdm(dataloader)
    for i, (input_ids_batch, attention_mask_batch, labels_batch, lengths_batch) in enumerate(train_bar):
        optimizer.zero_grad()
        logits = model(input_ids_batch.to(device), attention_mask_batch.to(device))
        loss = criterion(logits, labels_batch.float().to(device))
        loss.backward()
       
        optimizer.step()
        train_bar.set_description(f'Epoch {epoch}')
        total_loss += loss.item() *  len(labels_batch)
        total_samples += len(labels_batch)
       

        # compute the accuracy
        predictions = (logits > 0).long()
        correct_samples += (predictions == labels_batch.to(device)).sum().item()
        accuracy = correct_samples / total_samples
        average_loss = total_loss / total_samples
        train_bar.set_postfix({'loss': f'{average_loss}',  'accuracy': f'{accuracy}'})

    # evaluate the model
    model.eval()
    total_loss = 0
    total_samples = 0
    correct_samples = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels, lengths in eval_dataloader:
            # compute the model output
            logits = model(input_ids.to(device), attention_mask.to(device))
            # compute the loss
            loss = criterion(logits, labels.float().to(device))
            total_loss += loss.item()
            total_samples += len(labels)
            # compute the accuracy
            predictions = (logits > 0).long()
            correct_samples += (predictions == labels.to(device)).sum().item()
    accuracy = correct_samples / total_samples
    average_loss = total_loss / len(eval_dataloader)
    print(f'Accuracy: {accuracy}, Average Loss: {average_loss}', f'corretti: {correct_samples}')
    # keep the best model
    if average_loss < best_loss:
        best_loss = average_loss
        torch.save(model.state_dict(), 'best_model.pth')



# evaluate the model
model.eval()
total_loss = 0
total_samples = 0
correct_samples = 0
with torch.no_grad():
    for input_ids, attention_mask, labels, lengths in eval_dataloader:
        # compute the model output
        logits = model(input_ids.to('cuda'), attention_mask.to('cuda'))
        # compute the loss
        loss = criterion(logits, labels.float().to('cuda'))
        total_loss += loss.item()
        total_samples += len(labels)
        # compute the accuracy
        predictions = (logits > 0).long()
        correct_samples += (predictions == labels.to('cuda')).sum().item()
accuracy = correct_samples / total_samples
average_loss = total_loss / len(eval_dataloader)
print(f'Accuracy: {accuracy}, Average Loss: {average_loss}', f'corretti {correct_samples}')