import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, max_length=128)  # Reduce max_length

def generate_text(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs.input_ids,
        max_length=100,
        num_return_sequences=1,
        temperature=0.8,  
        top_k=30,        
        top_p=0.9,      
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def main():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    dataset = load_dataset("text", data_files={"train": "sample_text.txt"})
    
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

    prompt = "Once upon a time in a kingdom far away, there lived a brave knight."
    generated_text = generate_text(model, tokenizer, prompt)
    with open("generated_text.txt", "w", encoding="utf-8") as f:
        f.write(generated_text)

if __name__ == "__main__":
    main()
