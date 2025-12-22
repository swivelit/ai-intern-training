import argparse
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


def answer_question(model_dir, question, context):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
    model.eval()

    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=384,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    start = torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits) + 1

    answer = tokenizer.decode(inputs["input_ids"][0][start:end], skip_special_tokens=True)
    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Question Answering using fine-tuned BERT")
    parser.add_argument("--question", required=True, help="Question text")
    parser.add_argument("--context", required=True, help="Context paragraph")
    parser.add_argument("--model_dir", default="outputs/qa_model", help="Path to trained model")

    args = parser.parse_args()

    result = answer_question(args.model_dir, args.question, args.context)
    print("\nAnswer:", result)
