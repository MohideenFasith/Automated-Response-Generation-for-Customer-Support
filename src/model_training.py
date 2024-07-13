from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from data_preprocessing import load_and_preprocess_data
import torch

def train_model():
    # Load dataset
    dataset = load_and_preprocess_data()

    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize dataset
    def tokenize_function(examples):
        inputs = tokenizer(examples['query'], padding="max_length", truncation=True, max_length=50)
        outputs = tokenizer(examples['response'], padding="max_length", truncation=True, max_length=50)
        inputs['labels'] = outputs['input_ids']
        return inputs

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test']
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    train_model()
