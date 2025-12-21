# Project 6: Question Answering using BERT (DistilBERT)

## Dataset
- SQuAD v1.1 (loaded automatically using HuggingFace datasets)

## Setup
```bash
pip install -r requirements.txt
```

## Train Model
```bash
python train_qa.py
```

## Run Question Answering (CLI)
```bash
python run_qa.py "What is BERT?" "BERT is a transformer-based model developed by Google."
```

## Notes
- Uses DistilBERT for faster training on CPU.
- Training limited to 1 epoch and small batch size.