
import argparse
import torch
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering

parser = argparse.ArgumentParser(description="Run Question Answering")
parser.add_argument("--context", required=True)
parser.add_argument("--question", required=True)
args = parser.parse_args()

tokenizer = DistilBertTokenizerFast.from_pretrained("./qa_model")
model = DistilBertForQuestionAnswering.from_pretrained("./qa_model")

inputs = tokenizer(args.question, args.context, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

start = torch.argmax(outputs.start_logits)
end = torch.argmax(outputs.end_logits) + 1

answer = tokenizer.decode(inputs["input_ids"][0][start:end])
print("Answer:", answer)
