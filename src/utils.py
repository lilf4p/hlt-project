import json
import os
import pandas as pd

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

    # return train, dev and test dataset
    return df_train, df_dev, df_test

if __name__ == "__main__":
    # test load_ECHR
    df_train, df_dev, df_test = load_ECHR('ECHR_Dataset')
    print(df_train)
    print(df_dev)
    print(df_test)

    # test load_ECHR_small
    df_train, df_dev, df_test = load_ECHR_small('ECHR_Dataset',n=10)
    print(df_train)
    print(df_dev)
    print(df_test)


    
    


   


