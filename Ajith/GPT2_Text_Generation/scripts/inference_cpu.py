from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_PATH = "../gpt2-finetuned"

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)

prompt = input("Enter prompt: ")

inputs = tokenizer(prompt, return_tensors="pt")

output = model.generate(
    **inputs,
    max_length=100,
    do_sample=True,
    temperature=0.9,
    top_k=40,
    top_p=0.9
)

print("\nGenerated Text:\n")
print(tokenizer.decode(output[0], skip_special_tokens=True))
