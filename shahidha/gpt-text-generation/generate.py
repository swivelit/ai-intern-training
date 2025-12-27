from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_path = "./model"

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

prompt = input("Enter prompt: ")

inputs = tokenizer(prompt, return_tensors="pt")

output = model.generate(
    **inputs,
    max_length=150,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    do_sample=True
)

print("\nGenerated Text:\n")
print(tokenizer.decode(output[0], skip_special_tokens=True))
