from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset
dataset = load_dataset('text', data_files={'train': 'dataset.txt'})

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Tokenize
def tokenize_function(examples):
    tokens = tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=128
    )
    
    # IMPORTANT: add labels
    tokens["labels"] = tokens["input_ids"].copy()
    
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Load model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Training arguments (UPDATED for v5)
training_args = TrainingArguments(
    output_dir="./model",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_strategy="epoch",
    logging_steps=50
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train']
)

# Train
trainer.train()

# Save model
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")