import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import pandas as pd
from tqdm import tqdm
from bertclass import BertAttentionClassifier
from utils import collate_fn_chunks
from accelerate import Accelerator
import os
# examples settings
n_chunks = 8
batch_size = 2 # reduce batch size if u increase n_chunks for memory issues

# CUDA_VISIBLE_DEVICES = 0,2,3
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"
print('CUDA_VISIBLE_DEVICES:', os.environ["CUDA_VISIBLE_DEVICES"])

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

collate_fn = lambda x: collate_fn_chunks(x, n_chunks) # collate function with n. of chunks chosen
dataloader = torch.utils.data.DataLoader(dataset, batch_size= batch_size, shuffle=True, collate_fn=collate_fn)
eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# train the model with accelerate on GPU
accelerator = Accelerator()

# print accelerate config settings 
print(accelerator.state)

model = BertAttentionClassifier( bert_model_name='nlpaueb/legal-bert-base-uncased')
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-6)

model, optimizer, dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, dataloader, eval_dataloader
    )

num_epochs=4
update_epochs=int(16 / batch_size ) # optimizer.step() every 16 examples
best_loss = float('inf')

print('starting training...')
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_samples = 0
    correct_samples = 0
    train_bar = tqdm(dataloader)
    for i, (input_ids_batch, attention_mask_batch, labels_batch, lengths_batch) in enumerate(train_bar):
        logits = model(input_ids_batch, attention_mask_batch)
        loss = criterion(logits, labels_batch.float()) / update_epochs # compute the loss
        accelerator.backward(loss) 
        if ((i +1) % update_epochs ==0) or (i+1 ==len(dataloader)): 
            optimizer.step()
            optimizer.zero_grad()
        train_bar.set_description(f'Epoch {epoch}')
        total_loss += loss.item() *  len(labels_batch) * update_epochs # total running loss for visualizing it
        total_samples += len(labels_batch)
    

        # compute the accuracy
        predictions = (logits > 0).long()
        correct_samples += (predictions == labels_batch).sum().item()
        accuracy = correct_samples / total_samples
        average_loss = total_loss / total_samples
        train_bar.set_postfix({'loss': f'{average_loss = :.3f}',  'accuracy': f'{accuracy = :.3f}'})
    
    # evaluate the model
    model.eval()
    total_loss = 0
    total_samples = 0
    correct_samples = 0
    with torch.no_grad():
        eval_bar = tqdm(eval_dataloader)

        for input_ids, attention_mask, labels, lengths in eval_bar :
            # compute the model output
            logits = model(input_ids, attention_mask)
            # compute the loss
            loss = criterion(logits, labels.float()) 
            total_loss += loss.item()
            total_samples += len(labels)
            # compute the accuracy
            predictions = (logits > 0).long()
            correct_samples += (predictions == labels).sum().item()
    accuracy = correct_samples / total_samples
    average_loss = total_loss / len(eval_dataloader)
    print(f'Accuracy: {accuracy}, Average Loss: {average_loss}', f'corretti: {correct_samples}')
    # keep the best model
    if average_loss < best_loss:
        best_loss = average_loss
        torch.save(model.state_dict(), f'hier-legal-bert_{n_chunks}.pth')



