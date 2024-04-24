# imports
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
model = AutoModelForSequenceClassification.from_pretrained('models/google/bigbird-roberta-base', num_labels=2)
model
# load tokenized dataset
train_dataset_tokenized = Dataset.load_from_disk(f'../ECHR_Dataset_Tokenized/{model_name}/train')
dev_dataset_tokenized = Dataset.load_from_disk(f'../ECHR_Dataset_Tokenized/{model_name}/dev')
test_dataset_tokenized = Dataset.load_from_disk(f'../ECHR_Dataset_Tokenized/{model_name}/test')

# trainer
# CUDA_VISIBLE_DEVICES = 0,2,3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"

# load model
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

trainer.evaluate(test_dataset_tokenized)