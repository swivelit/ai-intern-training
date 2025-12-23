
import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def main(args):
    tokenizer = GPT2Tokenizer.from_pretrained("./model")
    model = GPT2LMHeadModel.from_pretrained("./model")

    inputs = tokenizer.encode(args.prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=150,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    args = parser.parse_args()
    main(args)
