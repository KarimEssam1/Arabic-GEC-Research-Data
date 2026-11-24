# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 20:50:45 2024

@author: Karim
"""

import os
import pandas as pd
from datasets import Dataset
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments,AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM, Trainer, TrainingArguments,GPT2TokenizerFast
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType,prepare_model_for_kbit_training
from transformers import TrainerCallback,GPT2Tokenizer,GPT2LMHeadModel
import matplotlib.pyplot as plt
from transformers import BartForCausalLM,BartTokenizer
import wandb
os.environ["WANDB_MODE"] = "disabled"


text_data=pd.read_csv('D:/Phd/Optimization/filtered_arabic_msa.csv')


# Convert pandas DataFrame to Hugging Face Dataset
text_data = Dataset.from_pandas(text_data)

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('aubmindlab/aragpt2-large')
model = GPT2LMHeadModel.from_pretrained('aubmindlab/aragpt2-large')

# Define preprocessing function
def preprocess_function(examples):
    inputs = examples['input']
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length")
    model_inputs["labels"] = model_inputs["input_ids"]
    return model_inputs

# Set pad_token to be the same as eos_token
tokenizer.pad_token = tokenizer.eos_token

# Verify that pad_token is set
assert tokenizer.pad_token == tokenizer.eos_token


# Apply preprocessing
tokenized_dataset = text_data.map(preprocess_function, batched=True)



# Split dataset into train and validation sets
train_test_data = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_data['train']
eval_dataset = train_test_data['test']


# Custom callback to collect logs
class LogCallback(TrainerCallback):
    def __init__(self):
        self.train_loss = []
        self.eval_loss = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if 'loss' in logs:
                self.train_loss.append((state.global_step, logs['loss']))
            if 'eval_loss' in logs:
                self.eval_loss.append((state.global_step, logs['eval_loss']))

# Initialize the callback
log_callback = LogCallback()


# Training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-text-generation",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=1,
    evaluation_strategy="epoch",  # Evaluate every epoch
    logging_dir='./logs',          # Directory for storing logs
    logging_steps=500,             # Log every 500 steps
    save_strategy="epoch",         # Save checkpoint every epoch
)

lora_config = LoraConfig(
    r=4,                      # Low-rank dimension
    lora_alpha=16,                 # Scaling factor
    lora_dropout=0.1,              # Dropout rate
    target_modules=["c_attn"],  # Target attention modules in BART
    #task_type="SEQ_2_SEQ_LM",
)

# Prepare the model for k-bit training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[log_callback]  # Add the custom callback
)


# Start training
trainer.train()
