
# Project 7: Text Generation using GPT-2 (Small)

## Overview
This project fine-tunes GPT-2 (small) on a custom text corpus and generates text from prompts.

## Dataset
Any free text corpus (news articles, Wikipedia subset, books, etc.)

## Setup
```bash
pip install transformers datasets torch
```

## Training
```bash
python train.py --data_file data/train.txt
```

## Inference
```bash
python generate.py --prompt "Artificial Intelligence is"
```
