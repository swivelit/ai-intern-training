
from transformers import pipeline

qa = pipeline("question-answering", model="./qa_model", tokenizer="./qa_model")

print("Enter context (paragraph):")
context = input()

print("Enter question:")
question = input()

result = qa(question=question, context=context)
print("\nAnswer:", result["answer"])
