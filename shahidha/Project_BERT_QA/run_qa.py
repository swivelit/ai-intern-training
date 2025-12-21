import argparse
from transformers import pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--question", required=True)
parser.add_argument("--context", required=True)
args = parser.parse_args()

qa = pipeline(
    "question-answering",
    model="./bert_qa_model",
    tokenizer="./bert_qa_model"
)

result = qa(question=args.question, context=args.context)

print("\nQuestion:", args.question)
print("Answer:", result["answer"])
print("Score:", round(result["score"], 3))
