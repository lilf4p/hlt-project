import sys
sys.path.append('../')
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from bertclass import BertAttentionClassifier
from utils import collate_fn_chunks
import seaborn as sns
import matplotlib.pyplot as plt


n_chunks = 4 # number of chunks to consider
batch_size = 4

device =torch.device('cuda', 2)
path_test ='ECHR_Dataset_Tokenized/legal-bert-base-uncased/df_test_tokenized.pkl'
df_test = pd.read_pickle(path_test)
documents_test = df_test[['input_ids', 'attention_mask', 'label']]
input_ids = documents_test.input_ids.tolist()
attention_mask = documents_test.attention_mask.tolist()
labels_test = documents_test.label.tolist()
input_ids = [torch.stack(i) for i in input_ids]
attention_mask = [torch.stack(i) for i in attention_mask]
input_ids =[torch.squeeze(i, dim=1) for i in input_ids]
attention_mask =[torch.squeeze(i, dim=1) for i in attention_mask]
lengths_test =[i.size(0) for i in input_ids]

class ECHRDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx], self.input_ids[idx].size(0) # last one is the length of the input_ids, used for padding


test_dataset = ECHRDataset(input_ids, attention_mask, labels_test)

collate_fn = lambda x: collate_fn_chunks(x, n_chunks) # collate function with n. of chunks chosen
dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

model = BertAttentionClassifier( bert_model_name='nlpaueb/legal-bert-base-uncased')
# load weights
path_weights = f'hier-legal-bert/best_model_{n_chunks}.pth'

# load a model that was saved previpously with accelerator, the model now has a 'module.' prefix in the state_dict keys
checkpoint = torch.load(path_weights, map_location=device)
# remove the 'module.' prefix from the state_dict keys
checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
model.load_state_dict(checkpoint)

model.to(device)

model.eval()
total_loss = 0
total_samples = 0
correct_samples = 0

with torch.no_grad():
    bar = tqdm(dataloader)
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_samples = 0
    correct_samples = 0
    total_pred = []
    labels_to_predict = []
    
    for input_ids, attention_mask, labels, lengths in bar:
        # compute the model output
        logits = model(input_ids.to(device), attention_mask.to(device))
        # compute the loss
        loss = criterion(logits, labels.float().to(device))            
        total_loss += loss.item()
        total_samples += len(labels)
        
        # compute the accuracy
        labels_to_predict.append(labels)
        predictions = (logits > 0).long()
        total_pred.append(predictions)
        correct_samples += (predictions == labels.to(device)).sum().item()

    accuracy = correct_samples / total_samples
    average_loss = total_loss / len(dataloader)
    
    print(f'Accuracy: {accuracy:.4f}, Average Loss: {average_loss:.4f}')
    total_pred = torch.cat(total_pred).cpu()
    labels_to_predict = torch.cat(labels_to_predict).cpu()
    print(classification_report(labels_to_predict, total_pred))

    # confusion matrix
    cm = confusion_matrix(labels_to_predict, total_pred)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
    plt.savefig(f'confusion_matrix_{n_chunks}.png')
