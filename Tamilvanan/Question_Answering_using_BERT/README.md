# Question Answering using BERT (CPU Friendly)

## Dataset
- SQuAD v1.1

## Model
- DistilBERT (lightweight, CPU compatible)

## Setup
```bash
pip install -r requirements.txt
```

## Training (small subset, CPU)
```bash
python train.py
```

## Run Question Answering from CLI
```bash
python qa_cli.py --question "Who developed BERT?" --context "BERT was developed by Google."
```