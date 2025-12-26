import argparse
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, required=True)
args = parser.parse_args()

tokenizer = GPT2Tokenizer.from_pretrained("./model")
model = GPT2LMHeadModel.from_pretrained("./model")

# Set pad token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# Encode input with attention mask
inputs = tokenizer(
    args.prompt,
    return_tensors="pt",
    padding=True
)

input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Generate text
outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_length=180,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.8,
    no_repeat_ngram_size=2
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
