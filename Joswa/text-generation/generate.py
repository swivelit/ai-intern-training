from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load trained model
model = GPT2LMHeadModel.from_pretrained('./model')
tokenizer = GPT2Tokenizer.from_pretrained('./model')

# Input
prompt = input("Enter prompt: ")

inputs = tokenizer.encode(prompt, return_tensors='pt')

# Generate
outputs = model.generate(
    inputs,
    max_length=100,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    do_sample=True
)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nGenerated Text:\n")
print(text)