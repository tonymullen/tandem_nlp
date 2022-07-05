from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments

from nlp import load_dataset
import torch
import numpy as np
import sys

from sklearn.metrics import precision_recall_fscore_support

# url = 'https://drive.google.com/uc?id=11_M4ootuT7I1G0RlihcC0cA3Elqotlc-'

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

if len(sys.argv) > 1:
    file_name = sys.argv[1]
else:
    file_name = 'english'

print(f"Processing {file_name} data")

data_file = f'/home/tonymullen/Dropbox/Research/Tandem_NLP/PreppedData/{file_name}.csv'

dataset = load_dataset('csv', data_files=data_file, split='train')
print(type(dataset))

dataset = dataset.train_test_split(test_size=0.3)
print(dataset)

train_set = dataset['train']
test_set = dataset['test']

# model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased').to('cuda')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

def preprocess(data):
    return tokenizer(data['text'], padding=True, truncation=True)

train_set = train_set.map(preprocess, batched=True, batch_size=len(train_set))
test_set = test_set.map(preprocess, batched=True, batch_size=len(test_set))

train_set.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_set.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

print(test_set)

batch_size = 4
epochs = 10

warmup_steps = 500
weight_decay = 0.01

training_args = TrainingArguments(
    output_dir='/media/tonymullen/Rapid/BERT/results_'+data_file,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=warmup_steps,
    weight_decay=weight_decay,
    # evaluate_during_training=True, # replace with following line
    evaluation_strategy='epoch',
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=test_set,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()

