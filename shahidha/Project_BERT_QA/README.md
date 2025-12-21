Project 6: Question Answering Using BERT

Dataset:
- SQuAD v1.1

Model:
- DistilBERT (distilbert-base-uncased)

Files:
- bert_qa.py : Training and evaluation
- run_qa.py  : Command-line inference
- bert_qa_model/ : Fine-tuned model

Steps to Run:
1. pip install -r requirements.txt
2. python bert_qa.py
3. python run_qa.py --question "..." --context "..."
