import json
import os
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
from sklearn.metrics import f1_score

def load_ECHR(path:str, anon:bool=False):

    """Load ECHR dataset from json to pandas dataframe
    
    Args:
        path (str): path to ECHR dataset
        anon (bool, optional): if True, load anonymized dataset. Defaults to False.
        
        Returns:
            df_train (pd.DataFrame): train dataset
            df_dev (pd.DataFrame): dev dataset
            df_test (pd.DataFrame): test dataset
            
    """

    # load train, dev and test dataset from json to pandas dataframe
    sfx = '_Anon' if anon else ''
    
    # define dataframe empty
    df_train = pd.DataFrame()
    df_dev = pd.DataFrame()
    df_test = pd.DataFrame()

    # train dataset
    train_path = path+'/EN_train'+sfx+'/'
    # for all the files in a directory 
    for filename in os.listdir(train_path):
        with open(train_path+filename) as f:
            # add json to dataframe as a row
            data = json.load(f)
            df = pd.DataFrame.from_dict(data, orient='index').T
            df_train = df_train._append(df, ignore_index=True)

    # dev dataset
    dev_path = path+'/EN_dev'+sfx+'/'
    # for all the files in a directory
    for filename in os.listdir(dev_path):
        with open(dev_path+filename) as f:
            # add json to dataframe as a row
            data = json.load(f)
            df = pd.DataFrame.from_dict(data, orient='index').T
            df_dev = df_dev._append(df, ignore_index=True)
    
    # test dataset
    test_path = path+'/EN_test'+sfx+'/'
    # for all the files in a directory
    for filename in os.listdir(test_path):
        with open(test_path+filename) as f:
            # add json to dataframe as a row
            data = json.load(f)
            df = pd.DataFrame.from_dict(data, orient='index').T
            df_test = df_test._append(df, ignore_index=True)

    # add a column with 0/1 labels to the dataframe 0 if VIOLATED_ARTICLE is empty, 1 otherwise
    df_train['LABEL'] = df_train['VIOLATED_ARTICLES'].apply(lambda x: 0 if x == [] else 1)
    df_dev['LABEL'] = df_dev['VIOLATED_ARTICLES'].apply(lambda x: 0 if x == [] else 1)
    df_test['LABEL'] = df_test['VIOLATED_ARTICLES'].apply(lambda x: 0 if x == [] else 1)

    # change name TEXT to text and LABEL to label
    df_train.rename(columns={'TEXT': 'text', 'LABEL': 'label'}, inplace=True)
    df_dev.rename(columns={'TEXT': 'text', 'LABEL': 'label'}, inplace=True)
    df_test.rename(columns={'TEXT': 'text', 'LABEL': 'label'}, inplace=True)

    train_text = df_train['text'].values
    dev_text = df_dev['text'].values
    test_text = df_test['text'].values

    df_train['text'] = [ "".join(x) for x in train_text]
    df_dev['text'] = [ "".join(x) for x in dev_text]
    df_test['text'] = [ "".join(x) for x in test_text]
    
    # return train, dev and test dataset
    return df_train, df_dev, df_test

def load_ECHR_small(path:str, anon:bool=False, n:int=100):

    """Load ECHR dataset from json to pandas dataframe

    Args:
        path (str): path to ECHR dataset
        anon (bool, optional): if True, load anonymized dataset. Defaults to False.
        n (int, optional): number of files to load. Defaults to 100.
    
    Returns:
        df_train (pd.DataFrame): train dataset
        df_dev (pd.DataFrame): dev dataset
        df_test (pd.DataFrame): test dataset

    """

    # load train, dev and test dataset from json to pandas dataframe
    sfx = '_Anon' if anon else ''
    
    # define dataframe empty
    df_train = pd.DataFrame()
    df_dev = pd.DataFrame()
    df_test = pd.DataFrame()

    i = 0

    # train dataset
    train_path = path+'/EN_train'+sfx+'/'
    # for all the files in a directory 
    for filename in os.listdir(train_path):

        if i == n:
            break

        with open(train_path+filename) as f:
            # add json to dataframe as a row
            data = json.load(f)
            df = pd.DataFrame.from_dict(data, orient='index').T
            df_train = df_train._append(df, ignore_index=True)
        
        i += 1

    i = 0

    # dev dataset
    dev_path = path+'/EN_dev'+sfx+'/'
    # for all the files in a directory
    for filename in os.listdir(dev_path):

        if i == n/2:
            break

        with open(dev_path+filename) as f:
            # add json to dataframe as a row
            data = json.load(f)
            df = pd.DataFrame.from_dict(data, orient='index').T
            df_dev = df_dev._append(df, ignore_index=True)

        i += 1
    
    i = 0

    # test dataset
    test_path = path+'/EN_test'+sfx+'/'
    # for all the files in a directory
    for filename in os.listdir(test_path):

        if i == n/2:
            break

        with open(test_path+filename) as f:
            # add json to dataframe as a row
            data = json.load(f)
            df = pd.DataFrame.from_dict(data, orient='index').T
            df_test = df_test._append(df, ignore_index=True)

        i += 1

    # add a column with 0/1 labels to the dataframe 0 if VIOLATED_ARTICLE is empty, 1 otherwise
    df_train['LABEL'] = df_train['VIOLATED_ARTICLES'].apply(lambda x: 0 if x == [] else 1)
    df_dev['LABEL'] = df_dev['VIOLATED_ARTICLES'].apply(lambda x: 0 if x == [] else 1)
    df_test['LABEL'] = df_test['VIOLATED_ARTICLES'].apply(lambda x: 0 if x == [] else 1)

    # return train, dev and test dataset
    return df_train, df_dev, df_test


# define a function which return a subsampling of the dataset, it has to be balanced
def subsampling(df, n:int=100):
    
        """Subsampling of the dataset, it has to be balanced
        
        Args:
            df (pd.DataFrame): dataset to subsample
            n (int, optional): number of files to load. Defaults to 100.
        
        Returns:
            df (pd.DataFrame): subsampled dataset
    
        """
    
        # define dataframe empty
        df_sub = pd.DataFrame()
    
        # define a list of the labels
        labels = df['LABEL'].unique()
    
        # for each label
        for label in labels:
            # select the rows of the label
            df_label = df[df['LABEL'] == label]
            # take n rows
            df_label = df_label.sample(n=n)
            # add to the dataframe
            df_sub = df_sub._append(df_label, ignore_index=True)
    
        # return the subsampled dataframe
        return df_sub

def metrics_model(dataset, model):

  predicted_labels = []
  labels = []

  for test in dataset:

    # convert to tensor
    test['input_ids'] = torch.Tensor(test['input_ids'])
    test['token_type_ids'] = torch.Tensor(test['token_type_ids'])
    test['attention_mask'] = torch.Tensor(test['attention_mask'])

    # reshape
    test['input_ids'] = test['input_ids'].reshape(1,-1).to(torch.int64)
    test['token_type_ids'] = test['token_type_ids'].reshape(1,-1).to(torch.int64)
    test['attention_mask'] = test['attention_mask'].reshape(1,-1).to(torch.int64)

    with torch.no_grad():
      logits = model(input_ids = test['input_ids'], token_type_ids = test['token_type_ids'], attention_mask = test['attention_mask']).logits

    predicted_class_id = logits.argmax().item()

    predicted_labels.append(predicted_class_id)
    labels.append(test['label'])
  
  return predicted_labels,labels

def k_fold_attention(model, criterion, optimizer, train_dataset, k_folds=5, epochs=10, batch_size=64):
    kf = KFold(n_splits=k_folds, shuffle=True)
    fold_results = []

    for fold, (train_indices, val_indices) in enumerate(kf.split(train_dataset), start=1):
        print(f"Fold {fold}/{k_folds}")

        # Initialize tqdm progress bar for epochs
        epoch_progress_bar = tqdm(range(epochs), desc="Epochs", unit="epoch")

        # Split dataset into train and validation for this fold
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler)

        best_val_loss = float('inf')
        best_model_state = None

        for epoch in epoch_progress_bar:
            model.train()
            running_loss = 0.0

            loop = tqdm(train_loader, total=len(train_loader), leave=False)

            for inputs, att_masks, labels in loop:
                optimizer.zero_grad()
                outputs = model(inputs, att_masks)
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)
            with torch.no_grad():
                model.eval()
                val_running_loss = 0.0

                for inputs, att_masks, labels in val_loader:
                    outputs = model(inputs, att_masks)
                    val_loss = criterion(outputs, labels.float())
                    val_running_loss += val_loss.item()

                val_loss = val_running_loss / len(val_loader)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict()

            # Update the description of the epoch progress bar
            epoch_progress_bar.set_postfix({
                "Train Loss": train_loss,
                "Val Loss": val_loss,
            })

        fold_results.append({
            'fold': fold,
            'best_val_loss': best_val_loss,
            'best_model_state': best_model_state
        })

    return fold_results

def k_fold_rnn(model, criterion, optimizer, train_dataset, k_folds=5, epochs=10, batch_size=64):
    kf = KFold(n_splits=k_folds, shuffle=True)
    fold_results = []

    for fold, (train_indices, val_indices) in enumerate(kf.split(train_dataset), start=1):
        print(f"Fold {fold}/{k_folds}")

        # Initialize tqdm progress bar for epochs
        epoch_progress_bar = tqdm(range(epochs), desc="Epochs", unit="epoch")

        # Split dataset into train and validation for this fold
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler)

        best_val_loss = float('inf')
        best_model_state = None

        for epoch in epoch_progress_bar:
            model.train()
            running_loss = 0.0

            loop = tqdm(train_loader, total=len(train_loader), leave=False)

            for inputs, att_masks, labels in loop:
                lengths = att_masks.sum(1).to('cpu')
                optimizer.zero_grad()
                outputs = model(inputs, lengths)
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)

            
            model.eval()
            with torch.no_grad():
                val_running_loss = 0.0

                for inputs, att_masks, labels in val_loader:
                    val_lengths = att_masks.sum(1).to('cpu')
                    outputs = model(inputs, val_lengths)
                    val_loss = criterion(outputs, labels.float())
                    val_running_loss += val_loss.item()

                val_loss = val_running_loss / len(val_loader)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict()

            # Update the description of the epoch progress bar
            epoch_progress_bar.set_postfix({
                "Train Loss": train_loss,
                "Val Loss": val_loss,
            })

        fold_results.append({
            'fold': fold,
            'best_val_loss': best_val_loss,
            'best_model_state': best_model_state
        })

    return fold_results

if __name__ == "__main__":
    # test load_ECHR
    df_train, df_dev, df_test = load_ECHR('ECHR_Dataset')
    print(df_train)
    print(df_dev)
    print(df_test)

    print(df_train['text'][0])

    # test subsampling
    #print(subsampling(df_train, n=10))
    #print(subsampling(df_dev, n=10))
    #print(subsampling(df_test, n=10))
 
    
    


   


