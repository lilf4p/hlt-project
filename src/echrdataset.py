from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import os

class ECHRDataset(Dataset):
    '''Dataset class for ECHR dataset.
    
    Args:
        data (list): List of tuples containing the data.
        attention_mask (list): List of attention masks.
        labels (list): List of labels.
    
    Returns:
        torch.Tensor: Data tensor.
        torch.Tensor: Attention mask tensor.
        torch.Tensor: Label tensor.
    '''
    def __init__(self, data, attention_mask, labels):
        self.data = data
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.attention_mask[idx], self.labels[idx]
