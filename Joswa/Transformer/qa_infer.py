import sys
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_path = "qa_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)

def answer_question(question, context):
    inputs = tokenizer(question, context, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # Get best start and end
    start = torch.argmax(start_logits)
    end = torch.argmax(end_logits)

    # 🔥 FIX: ensure valid span
    if end < start:
        end = start

    answer_ids = inputs["input_ids"][0][start:end+1]

    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)

    return answer.strip()

if __name__ == "__main__":
    question = sys.argv[1]
    context = sys.argv[2]

    result = answer_question(question, context)
    print("\nAnswer:", result)