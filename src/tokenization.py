import torch
import numpy as np
from utils import load_ECHR
import transformers
import pandas as pd
from tqdm import tqdm
def tokenize_document(row, tokenizer, max_length=512):
    '''
        Tokenize a document using the provided tokenizer
        input:
            document: iterable, document to be tokenized. Every element of the iterable is a sentence
            tokenizer: transformers tokenizer
            max_length: int, maximum length of the tokenized document
    '''
    # tokenize the document in chunks of max_length
    tokenized_document = []

    for sentence in row:
        tokenized_sentence = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=max_length, pad_to_max_length=False, return_tensors='pt', return_attention_mask=True).input_ids
        # if the tokenized doc is not empty, try to add the sentence to the last tokenized sentence
        if tokenized_document:
            if tokenized_document[-1].size()[1] + tokenized_sentence.size()[1] <= max_length:
                tokenized_document[-1] = torch.cat((tokenized_document[-1], tokenized_sentence[0:,1:]), dim=1)
            else:
                tokenized_document.append(tokenized_sentence)
        else:
            tokenized_document.append(tokenized_sentence)
    # now we add attention_masks and token_type_ids
    # pad every sentence to max_length
    for i, sentence in enumerate(tokenized_document):
        tokenized_document[i] = torch.cat((sentence, torch.zeros((1, max_length - sentence.size()[1]), dtype=torch.long)), dim=1)
    return tokenized_document


def add_attention_masks(row):
    '''
        Add attention masks to the tokenized document
    '''
    attention_masks = []
    for sentence in row:
        attention_mask = torch.ones_like(sentence)
        attention_mask[sentence==0] = 0
        attention_masks.append(attention_mask)
    return attention_masks

def tokenization_pipeline(tokenizer, path, max_length=512, anon=False, save=False):
    print('loading dataset')
    df_train = pd.read_csv('/storagenfs/l.stoppani/hlt-project/hlt-project/ECHR_Dataset/train.csv')
    df_dev = pd.read_csv('/storagenfs/l.stoppani/hlt-project/hlt-project/ECHR_Dataset/dev.csv')
    df_test = pd.read_csv('/storagenfs/l.stoppani/hlt-project/hlt-project/ECHR_Dataset/test.csv')
    # tokenize the documents
    print('tokenizing')
    tqdm.pandas()

    df_train['input_ids'] = df_train['text'].progress_apply(lambda row: tokenize_document(row, tokenizer, max_length=max_length))
    df_dev['input_ids'] = df_dev['text'].progress_apply(lambda row: tokenize_document(row, tokenizer, max_length=max_length))
    df_test['input_ids'] = df_test['text'].progress_apply(lambda row: tokenize_document(row, tokenizer, max_length=max_length))
    # add attention masks
    df_train['attention_mask'] = df_train['input_ids'].progress_apply(lambda row: add_attention_masks(row))
    df_dev['attention_mask'] = df_dev['input_ids'].progress_apply(lambda row: add_attention_masks(row))
    df_test['attention_mask'] = df_test['input_ids'].progress_apply(lambda row: add_attention_masks(row))
    if save:
        df_train.to_pickle(path + '/train_tokenized.pkl')
        df_dev.to_pickle(path + '/dev_tokenized.pkl')
        df_test.to_pickle(path + '/test_tokenized.pkl')
    return df_train, df_dev, df_test

if __name__ == '__main__':
    tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')

    tokenization_pipeline(tokenizer, '../ECHR_Dataset_Tokenized/distilbert-base-uncased', max_length=512, anon=False, save=True)


