import argparse
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForQuestionAnswering,
    pipeline
)

# -----------------------------
# Argument parsing
# -----------------------------
parser = argparse.ArgumentParser(description="QA using fine-tuned DistilBERT")
parser.add_argument("--question", required=True, help="Question string")
parser.add_argument("--context", required=True, help="Context paragraph")
args = parser.parse_args()

# -----------------------------
# Load model & tokenizer
# -----------------------------
model_path = "./model"

tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForQuestionAnswering.from_pretrained(model_path)

# -----------------------------
# QA pipeline
# -----------------------------
qa_pipeline = pipeline(
    "question-answering",
    model=model,
    tokenizer=tokenizer
)

# -----------------------------
# Inference
# -----------------------------
result = qa_pipeline(
    question=args.question,
    context=args.context
)

print("\nAnswer:", result["answer"])
print("Score :", round(result["score"], 4))
