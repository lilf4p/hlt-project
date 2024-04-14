from torch.utils.data import Dataset
import torch.nn.functional as F
from echrdataset import ECHRDataset
import torch
import numpy as np
import pandas as pd
import json
import os

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

def reset_weights(m):
        '''
            Try resetting model weights to avoid
            weight leakage.
        '''
        for layer in m.children():
            if hasattr(layer, 'reset_parameters'):
                print(f'Reset trainable parameters of layer = {layer}')
                layer.reset_parameters()

def get_device():
    """
    Get the device to run the model on.
    Returns:
        torch.device: The device to run the model on.
    """
    if (torch.cuda.is_available()):
        print("Running on GPU")
        device = torch.device('cuda', 1)
    elif (torch.backends.mps.is_available()):
        print("Running on MPS")
        device = torch.device('mps')
    else :
        print("Running on CPU")
        device = torch.device('cpu')
    return device

# pad the data to be of the same shape
def __pad_data__(data, max_len):
    padded_data = []
    attention_masks = []
    for i in range(len(data)):
        attention_masks.append([1] * data[i].shape[0] + [0] * (max_len - data[i].shape[0]))
        padded_data.append(F.pad(data[i], (0, 0, 0, max_len - data[i].shape[0])))
    #print(len(attention_masks))
    return torch.stack(padded_data), torch.tensor(attention_masks)
    
def create_dataset(path_to_datasets):
    
    # load data
    train = torch.load('embeddings/legal-bert-base-uncased/emb_tr_cpu.pkl')
    dev = torch.load('embeddings/legal-bert-base-uncased/emb_dev_cpu.pkl')
    test = torch.load('embeddings/legal-bert-base-uncased/emb_test_cpu.pkl')

    print('Train '+str(len(train)),'Dev '+str(len(dev)), 'Test '+str(len(test)))
    
    # concat dev to train series
    train = np.concatenate((train, dev))

    print('Train + Dev = '+str(len(train)))

    # load labels
    train_labels = pd.read_pickle('embeddings/legal-bert-base-uncased/train_labels.pkl')
    dev_labels = pd.read_pickle('embeddings/legal-bert-base-uncased/dev_labels.pkl')
    test_labels = pd.read_pickle('embeddings/legal-bert-base-uncased/test_labels.pkl')

    # concat dev labels to train labels
    train_labels = torch.tensor(np.concatenate((train_labels, dev_labels)))

    # pad the data
    max_len_train = max([x.shape[0] for x in train])
    max_len_test = max([x.shape[0] for x in test])
    train, train_attention_masks = __pad_data__(train, max_len_train)
    test, test_attention_masks = __pad_data__(test, max_len_test)

    # create the datasets
    train_dataset = ECHRDataset(train, train_attention_masks, train_labels)
    test_dataset = ECHRDataset(test, test_attention_masks, test_labels)

    print (train_dataset.data.device)

    # save the datasets
    if not os.path.exists(path_to_datasets):
        os.makedirs(path_to_datasets)
    torch.save(train_dataset, path_to_datasets+'train_dataset.pt')
    torch.save(test_dataset, path_to_datasets+'test_dataset.pt')

def load_dataset(path_to_datasets):
    train_dataset = torch.load(path_to_datasets+'train_dataset.pt')
    test_dataset = torch.load(path_to_datasets+'test_dataset.pt')
    print(len(train_dataset))
    return train_dataset, test_dataset


def collate_fn_chunks(data, max_chunks=3):
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