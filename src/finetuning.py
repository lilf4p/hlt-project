# CUDA_VISIBLE_DEVICES = 0,2,3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"
print(os.environ["CUDA_VISIBLE_DEVICES"])

import numpy as np
import transformers
import torch
import pandas as pd
from transformers import Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from datasets import Dataset

# load longformer
model_name ='google/bigbird-roberta-base'
from transformers import AutoModelForSequenceClassification, BigBirdTokenizerFast
tokenizer = BigBirdTokenizerFast.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
tokenizer.is_fast 
tokenize =False
if tokenize:
    def encode_text(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)

    df_train= pd.read_csv('../ECHR_Dataset/train.csv')
    df_test= pd.read_csv('../ECHR_Dataset/test.csv')         
    df_dev= pd.read_csv('../ECHR_Dataset/dev.csv')


    df_train = df_train[['text','label']]
    df_test = df_test[['text','label']]
    df_dev = df_dev[['text','label']]

    # convert to huggingface dataset    
    from datasets import Dataset
    train_dataset = Dataset.from_pandas(df_train)
    dev_dataset = Dataset.from_pandas(df_dev)
    test_dataset = Dataset.from_pandas(df_test)
    print(train_dataset)
    train_dataset_tokenized = train_dataset.map(encode_text, batched=True)
    dev_dataset_tokenized = dev_dataset.map(encode_text, batched=True)
    test_dataset_tokenized = test_dataset.map(encode_text, batched=True)
    # save tokenized dataset
    train_dataset_tokenized.save_to_disk(f'../ECHR_Dataset_Tokenized/{model_name}/train')
    dev_dataset_tokenized.save_to_disk(f'../ECHR_Dataset_Tokenized/{model_name}/dev')
    test_dataset_tokenized.save_to_disk(f'../ECHR_Dataset_Tokenized/{model_name}/test')

else:
    # load tokenized dataset
    train_dataset_tokenized = Dataset.load_from_disk(f'../ECHR_Dataset_Tokenized/{model_name}/train')
    dev_dataset_tokenized = Dataset.load_from_disk(f'../ECHR_Dataset_Tokenized/{model_name}/dev')
    test_dataset_tokenized = Dataset.load_from_disk(f'../ECHR_Dataset_Tokenized/{model_name}/test')

# trainer
# CUDA_VISIBLE_DEVICES = 0,2,3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"
print(os.environ["CUDA_VISIBLE_DEVICES"])

path ='models/'+ model_name

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,

        'recall': recall
    }


training_args = TrainingArguments(
    output_dir=path,                 # output directory
    learning_rate=3e-6,              # learning rate
    num_train_epochs=2,              # total number of training epochs
    per_device_train_batch_size=1,  # batch size per device during training
    per_device_eval_batch_size=2,   # batch size for evaluation
    fp16=True,
    weight_decay=0.01,               # strength of weight decay
    logging_dir=path+'/log',         # directory for storing logs
    evaluation_strategy='steps',
    eval_steps=100,
    save_steps=100,
    
    logging_steps=100,
    save_total_limit=2,
    gradient_accumulation_steps = 5,
    load_best_model_at_end=True,
    metric_for_best_model = 'eval_loss',
    seed = 42
)

data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,                                     # the instantiated 🤗 Transformers model to be trained
    args=training_args,                              # training arguments, defined above
    train_dataset=train_dataset_tokenized,           # training dataset
    eval_dataset =dev_dataset_tokenized,             # evaluation dataset
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()
pred = trainer.predict(test_dataset_tokenized)

trainer.save_model()
