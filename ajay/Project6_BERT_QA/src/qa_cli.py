import argparse
from transformers import pipeline

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, default="outputs/qa_model", help="Path to saved fine-tuned model")
    p.add_argument("--context", type=str, required=True, help="Context passage")
    p.add_argument("--question", type=str, required=True, help="Question to ask")
    return p.parse_args()

def main():
    args = parse_args()
    qa = pipeline("question-answering", model=args.model_dir, tokenizer=args.model_dir)
    out = qa(question=args.question, context=args.context)
    print("Answer:", out.get("answer"))
    print("Score:", out.get("score"))
    print("Start:", out.get("start"), "End:", out.get("end"))

if __name__ == "__main__":
    main()
