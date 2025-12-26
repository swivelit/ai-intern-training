
# Project 7: Text Generation using GPT-2 (Small)

## Overview
This project demonstrates fine-tuning GPT-2 (small) on a custom text corpus and generating text based on prompts.

## Dataset
Use any free text corpus (Wikipedia subset, news articles, etc.).
Place your dataset as a `.txt` file in the `data/` folder.

## Setup
```bash
pip install -r requirements.txt
```

## Training
```bash
python train.py
```

## Inference
```bash
python infer.py --prompt "Artificial Intelligence is"
```

## Output
- Fine-tuned GPT-2 model
- Generated paragraphs from prompts
