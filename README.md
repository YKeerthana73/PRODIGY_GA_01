# Fine-Tuning GPT-2 for Text Generation

Welcome to the Fine-Tuning GPT-2 for Text Generation project! This repository demonstrates how to fine-tune the GPT-2 model to generate coherent and contextually relevant text based on given prompts. Using GPT-2, a transformer model developed by OpenAI, you will learn how to fine-tune the model on a custom dataset to create text that mimics the style and structure of your training data.

## Introduction

Text generation is a powerful application of deep learning, allowing models to generate text based on given prompts. In this project, we use GPT-2, a state-of-the-art language model known for producing fluent and contextually appropriate text. By leveraging the Hugging Face Transformers library, we can easily fine-tune GPT-2 for specific tasks.

GPT-2 has various applications, including:
- Creative writing assistance
- Automated content generation
- Chatbot development
- Text completion and summarization

## Features

- **Custom Dataset Fine-Tuning**: Train GPT-2 on your custom dataset to generate text in a specific style.
- **Text Generation**: Generate text based on prompts using the fine-tuned model.
- **CUDA Support**: Leverage GPU acceleration (if available) for faster and more efficient training and text generation.

## Prerequisites

- Install Python 3.7 or higher from the [official website](https://www.python.org/downloads/).

## Setup

### 1. Create a folder and activate a virtual environment

First, create a folder for the project and set up a Python virtual environment to manage dependencies:

```sh
# Create folder
mkdir gpt2-finetuning

# Navigate to folder
cd gpt2-finetuning

# Create virtual environment
python -m venv gpt2-env

# Activate virtual environment (Windows)
gpt2-env\Scripts\activate

# Activate virtual environment (macOS/Linux)
source gpt2-env/bin/activate
```

### 2. Install the required packages
Next, install the necessary Python packages: torch, transformers, and datasets.
```
pip install torch transformers datasets
```

## Code Explanation
Main Script: train_gpt2.py  
```
import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, max_length=128)

def main():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    dataset = load_dataset("text", data_files={"train": "your_dataset.txt"})
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)  
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    
    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples
    
    tokenized_datasets = tokenized_datasets.map(add_labels, batched=True)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        learning_rate=3e-5,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=1000,
        eval_strategy="no",
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
    )

    trainer.train()

if __name__ == "__main__":
    main()

```
## Functions and Implementation Details
- Tokenization: Tokenizes input text using GPT-2 tokenizer with padding and truncation for model compatibility.
- Model Training: Fine-tunes the GPT-2 model on a custom dataset to generate coherent text based on the input prompt.
- CUDA Support: Utilizes GPU acceleration for faster training if a CUDA-compatible GPU is available.

Customization
Dataset: Replace `your_dataset.txt` with your own dataset file containing text data for training.
Training Parameters: Adjust `num_train_epochs`, `learning_rate`, and `per_device_train_batch_size` in `TrainingArguments` for optimal model performance based on your specific requirements.


## Contributing
Contributions to improve this project are welcome! If you'd like to contribute:

- Fork the repository.
- Create a new branch with a descriptive name (`git checkout -b my-branch-name`).
- Make your changes and commit them (`git commit -am 'Add some feature'`).
- Push to the branch (`git push origin my-branch-name`).
- Create a new Pull Request.
