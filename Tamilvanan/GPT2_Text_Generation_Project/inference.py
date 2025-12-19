from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_path = "./model"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

prompt = input("Enter prompt: ")
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_length=150,
    do_sample=True,
    top_k=50,
    top_p=0.95
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
