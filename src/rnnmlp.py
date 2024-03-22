import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
from torch.utils.data import TensorDataset
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from copy import deepcopy
from src.utils import reset_weights
import time

class RnnMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, hidden_dim_mlp, output_dim, dropout=0, bidirectional=False):
        super(RnnMLP, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim, hidden_dim_mlp)
        self.fc2 = nn.Linear(hidden_dim_mlp, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x, lengths):
        # pack the padded sequence
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        # forward pass through LSTM
        out, (h_n, c_n) = self.rnn(x)
        # unpack the packed sequence
        output= self.fc(h_n[-1])
        output=self.relu(output)
        output= self.dropout(output)
        output = self.fc2(output)
        output=self.sigmoid(output)
        return output.flatten()

    def k_fold(self, criterion, train_dataset, lr=0.001, weight_decay=1e-4, k_folds=5, epochs=50, batch_size=64):

        train_dataset.attention_mask = train_dataset.attention_mask.to('cpu')

        kf = KFold(n_splits=k_folds, shuffle=True)

        fold_results = {}

        for fold, (train_indices, val_indices) in enumerate(kf.split(train_dataset), start=1):
            print(f"Fold {fold}/{k_folds}")

            reset_weights(self)
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

            fold_results[fold] = {}

            # Initialize tqdm progress bar for epochs
            epoch_progress_bar = tqdm(range(epochs), desc="Epochs", unit="epoch")

            # Split dataset into train and validation for this fold
            train_sampler = SubsetRandomSampler(train_indices)
            val_sampler = SubsetRandomSampler(val_indices)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
            val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler)

            best_val_loss = float('inf')
            best_val_acc = 0.0
            best_val_f1 = 0.0

            fold_results[fold]['stats'] = {
                'train_losses': [],
                'val_losses': [],
                'val_accs': [],
                'val_f1s': []
            }

            for epoch in epoch_progress_bar:
                self.train()
                running_loss = 0.0

                loop = tqdm(train_loader, total=len(train_loader), leave=False)

                for inputs, att_masks, labels in loop:
                    lengths = att_masks.sum(1)
                    optimizer.zero_grad()
                    outputs = self(inputs, lengths)
                    loss = criterion(outputs, labels.float())
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                train_loss = running_loss / len(train_loader)

                
                self.eval()
                with torch.no_grad():
                    val_running_loss = 0.0

                    predictions = []
                    labels_topredict = []

                    for inputs, att_masks, labels in val_loader:
                        val_lengths = att_masks.sum(1)
                        outputs = self(inputs, val_lengths)
                        val_loss = criterion(outputs, labels.float())
                        val_running_loss += val_loss.item()
                        predictions.append(torch.round(outputs).to('cpu'))
                        labels_topredict.append(labels.to('cpu'))

                    predictions = torch.cat(predictions).numpy()
                    labels_topredict = torch.cat(labels_topredict).numpy()
                    val_loss = val_running_loss / len(val_loader)
                    val_acc = accuracy_score(labels_topredict, predictions)
                    val_f1 = f1_score(labels_topredict, predictions, average='weighted')

                    fold_results[fold]['stats']['train_losses'].append(train_loss)
                    fold_results[fold]['stats']['val_losses'].append(val_loss)
                    fold_results[fold]['stats']['val_accs'].append(val_acc)
                    fold_results[fold]['stats']['val_f1s'].append(val_f1)

                    # If the validation loss is the best, save the validation 
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_val_acc = val_acc
                        best_val_f1 = val_f1

                # Update the description of the epoch progress bar
                epoch_progress_bar.set_postfix({
                    "Train Loss": train_loss,
                    "Val Loss": val_loss,
                    "Val Acc": val_acc,
                    "Val F1": val_f1
                })

            fold_results[fold]['result'] = {
                'fold': fold,
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
                'best_val_f1': best_val_f1
            }

        return fold_results
    
    def train_loop(self, criterion, train_dataset, lr=0.0001, weight_decay=0.001, epochs=50, batch_size=64):
    
        train_dataset.attention_mask = train_dataset.attention_mask.to('cpu')

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_loss = float('inf')
        best_val_acc = 0.0
        best_val_f1 = 0.0

        train_progress_bar = tqdm(range(epochs), desc="Epochs", unit="epoch")

        dataset_size = len(train_dataset)
        val_size = int(0.05 * dataset_size)
        train_size = dataset_size - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        for epoch in train_progress_bar:
            self.train()
            running_loss = 0.0

            loop = tqdm(train_loader, total=len(train_loader), leave=False)

            for inputs, att_masks, labels in loop:
                lengths = att_masks.sum(1)
                optimizer.zero_grad()
                outputs = self(inputs, lengths)
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)

            with torch.no_grad():
                self.eval()
                val_running_loss = 0.0
                predictions = []
                labels_topredict = []
                
                for inputs, att_masks, labels in val_loader:
                    val_lengths = att_masks.sum(1)
                    outputs = self(inputs, val_lengths)
                    val_loss = criterion(outputs, labels.float())
                    val_running_loss += val_loss.item()
                    predictions.append(torch.round(outputs).to('cpu'))
                    labels_topredict.append(labels.to('cpu'))
                
                predictions = torch.cat(predictions).numpy()
                labels_topredict = torch.cat(labels_topredict).numpy()
                val_loss = val_running_loss / len(val_loader)
                val_acc = accuracy_score(labels_topredict, predictions)
                val_f1 = f1_score(labels_topredict, predictions, average='weighted')

                # If the validation loss is the best, save the validation 
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    best_val_f1 = val_f1
                    best_model_state_dict = deepcopy(self.state_dict())

                # Update the description of the epoch progress bar
                train_progress_bar.set_postfix({
                    "Train Loss": train_loss,
                    "Val Loss": val_loss,
                    "Val Acc": val_acc,
                    "Val F1": val_f1,
                })

        return best_model_state_dict, best_val_loss, best_val_acc, best_val_f1