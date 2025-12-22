# Transformer Project 6: Question Answering using BERT

## Objective
Fine-tune a transformer-based Question Answering model using the SQuAD v1.1 dataset.

## Dataset
- Stanford Question Answering Dataset (SQuAD v1.1)
- https://rajpurkar.github.io/SQuAD-explorer/

## Model
- DistilBERT (CPU-friendly)
- Can be replaced with bert-base-uncased if GPU is available

## Installation
```bash
pip install -r requirements.txt
```

## Training
```bash
python train.py
```

## Run QA (Command Line)
```bash
python qa_cli.py \
 --question "What is BERT?" \
 --context "BERT is a transformer-based language model developed by Google."
```

## Notes
- Training uses a reduced dataset subset for faster execution
- Suitable for systems without GPU
- Ideal for academic submission
