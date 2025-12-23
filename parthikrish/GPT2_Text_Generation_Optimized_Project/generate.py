
import argparse
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

parser = argparse.ArgumentParser(description="GPT-2 Text Generation")
parser.add_argument("--prompt", required=True)
parser.add_argument("--max_length", type=int, default=120)
args = parser.parse_args()

tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_model")
model = GPT2LMHeadModel.from_pretrained("./gpt2_model")

inputs = tokenizer.encode(args.prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_length=args.max_length,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
