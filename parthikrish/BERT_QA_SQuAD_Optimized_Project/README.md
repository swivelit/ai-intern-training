
# Question Answering using BERT (SQuAD v1.1)

## Dataset
SQuAD v1.1
https://rajpurkar.github.io/SQuAD-explorer/

## Model
- DistilBERT (CPU/GPU friendly)
- Can be replaced with bert-base-uncased if GPU available

## Install Requirements
pip install torch transformers datasets

## Train Model
python train_qa.py

## Run Question Answering (CLI)
python run_qa.py --context "Paris is the capital of France." --question "What is the capital of France?"

## Notes
- Training limited to small samples for low-resource systems
- Output model saved in ./qa_model/
