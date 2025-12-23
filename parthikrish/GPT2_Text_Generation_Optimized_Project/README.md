
# Text Generation using GPT-2 (Small)

## Description
Fine-tune GPT-2 small on a custom text corpus and generate text from prompts.

## Dataset
- Any free text corpus (Wikipedia subset, news articles, etc.)
- For demo, HuggingFace 'wikitext-2' is used

## Requirements
pip install torch transformers datasets

## Train Model
python train_gpt2.py

## Generate Text
python generate.py --prompt "Artificial Intelligence will"

## Notes
- GPT-2 small (124M parameters)
- Training optimized for CPU / low-resource systems
