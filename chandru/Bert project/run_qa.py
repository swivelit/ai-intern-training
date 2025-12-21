import sys
from transformers import pipeline

if len(sys.argv) < 3:
    print("Usage: python run_qa.py <question> <context>")
    sys.exit(1)

question = sys.argv[1]
context = sys.argv[2]

qa_pipeline = pipeline("question-answering", model="./model", tokenizer="./model")

result = qa_pipeline(question=question, context=context)
print("Answer:", result["answer"])
print("Score:", result["score"])