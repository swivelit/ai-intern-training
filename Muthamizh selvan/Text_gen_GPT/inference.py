from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import os

# Load Fine-Tuned Model
model_path = "./fine_tuned_model"

if not os.path.exists(model_path):
    print(f"Error: Model not found in {model_path}. Please run train.py first.")
    exit()

# Use base distilgpt2 tokenizer as it's more reliable
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained(model_path)

tokenizer.pad_token = tokenizer.eos_token
model.eval()

# Prompt Input
prompt = input("Enter your prompt: ")
if not prompt.strip():
    prompt = "Artificial intelligence is"

inputs = tokenizer(prompt, return_tensors="pt")

print("\nGenerating Text...\n")

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id
    )

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated Text:\n")
print(generated_text)
