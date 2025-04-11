from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer

from datasets import load_dataset

import numpy as np

import evaluate

raw_datasets = load_dataset("glue", "mrpc")

checkpoint = 'bert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example: dict) -> dict:
    return tokenizer(example['sentence1'], example['sentence2'], truncation=True)

tokenized_datasets = raw_datasets.map(lambda b: tokenizer(b['sentence1'], b['sentence2'], truncation=True), batched=True)
# print(tokenized_datasets)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments('test-trainer', eval_strategy='epoch')
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

def compute_metrics(eval_preds):
    metric = evaluate.load('glue", "mrpc')
    logits, labels = eval_preds
    
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

