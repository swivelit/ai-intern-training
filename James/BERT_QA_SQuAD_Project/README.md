
# BERT Question Answering (SQuAD v1.1)

## Project Overview
Fine-tune a BERT model for Question Answering using the SQuAD v1.1 dataset.

## Setup
```bash
pip install transformers datasets torch
```

## Train Model
```bash
python train.py
```

## Ask Questions from CLI
```bash
python qa_cli.py --question "Who wrote the Constitution?" --context "The Constitution was written by James Madison."
```
