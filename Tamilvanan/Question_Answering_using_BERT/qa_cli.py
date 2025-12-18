import argparse
from transformers import pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--question", required=True)
parser.add_argument("--context", required=True)
args = parser.parse_args()

qa = pipeline("question-answering", model="qa_model", tokenizer="qa_model")
result = qa(question=args.question, context=args.context)

print("Answer:", result["answer"])