import json
import os
import pandas as pd

def load_ECHR(anon:bool=False):

    # load train, dev and test dataset from json to pandas dataframe
    sfx = '_Anon' if anon else ''
    
    # define dataframe empty
    df_train = pd.DataFrame()
    df_dev = pd.DataFrame()
    df_test = pd.DataFrame()

    # train dataset
    train_path = 'ECHR_Dataset/EN_train'+sfx+'/'
    # for all the files in a directory 
    for filename in os.listdir(train_path):
        with open(train_path+filename) as f:
            # add json to dataframe as a row
            data = json.load(f)
            df = pd.DataFrame.from_dict(data, orient='index').T
            df_train = df_train._append(df, ignore_index=True)

    # dev dataset
    dev_path = 'ECHR_Dataset/EN_dev'+sfx+'/'
    # for all the files in a directory
    for filename in os.listdir(dev_path):
        with open(dev_path+filename) as f:
            # add json to dataframe as a row
            data = json.load(f)
            df = pd.DataFrame.from_dict(data, orient='index').T
            df_dev = df_dev._append(df, ignore_index=True)
    
    # test dataset
    test_path = 'ECHR_Dataset/EN_test'+sfx+'/'
    # for all the files in a directory
    for filename in os.listdir(test_path):
        with open(test_path+filename) as f:
            # add json to dataframe as a row
            data = json.load(f)
            df = pd.DataFrame.from_dict(data, orient='index').T
            df_test = df_test._append(df, ignore_index=True)

    # return train, dev and test dataset
    return df_train, df_dev, df_test
            

if __name__ == "__main__":
    # test load_ECHR
    df_train, df_dev, df_test = load_ECHR()
    print(df_train)
    print(df_dev)
    print(df_test)

    
    


   


