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

class AttentionMLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout=0, weight_decay=0.01):
        super(AttentionMLP, self).__init__()
        # vector for query attention
        self.selector = nn.parameter.Parameter(torch.randn(input_dim, 1))
        self.Value= nn.Linear(input_dim, input_dim, bias=False)
        self.Key = nn.Linear(input_dim, input_dim, bias=False)
        # mlp layers
        layers = []
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_sizes[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, attention_mask=None):
        # attention
        key = self.Key(x)

        value = self.Value(x)

        non_normalized_attention = torch.matmul(key, self.selector)
        if attention_mask is not None:
            attention_mask=attention_mask.unsqueeze(2)

            non_normalized_attention = non_normalized_attention.masked_fill(attention_mask == 0, -1e9)
        attention = F.softmax(non_normalized_attention, dim=1)
        # permute the attention to match the shape of the value
        attention = attention.permute(0, 2, 1)

        x = torch.matmul(attention, value)

        # mlp
        x = self.mlp(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x.flatten()

    def k_fold(self, criterion, train_dataset, lr=0.0001, weight_decay=0.001, k_folds=5, epochs=50, batch_size=64):
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
                    optimizer.zero_grad()
                    outputs = self(inputs, att_masks)
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
                        outputs = self(inputs, att_masks)
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
                    "Val F1": val_f1,
                })

            fold_results[fold]['result'] = {
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
                'best_val_f1': best_val_f1, 
            }

        return fold_results

    def train_loop(self, criterion, train_dataset, lr=0.0001, weight_decay=0.001, epochs=50, batch_size=64):
    
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_loss = float('inf')
        best_val_acc = 0.0
        best_val_f1 = 0.0

        train_progress_bar = tqdm(range(epochs), desc="Epochs", unit="epoch")

        dataset_size = len(train_dataset)
        val_size = int(0.1 * dataset_size)
        train_size = dataset_size - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        # from subset to dataset pandas
        #print(len(val_dataset))
        #check_labels = val_dataset[:][2]
        #print(check_labels.sum())

        # TODO: MAKE THE VALIDATION UNBALANCED 66% 1   
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        for epoch in train_progress_bar:
            self.train()
            running_loss = 0.0

            loop = tqdm(train_loader, total=len(train_loader), leave=False)

            for inputs, att_masks, labels in loop:
                optimizer.zero_grad()
                outputs = self(inputs, att_masks)
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
                    outputs = self(inputs, att_masks)
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