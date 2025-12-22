# Text Generation using GPT-2 (Small) – CPU Optimized

This project is optimized for CPU-only and low-RAM systems.

## Optimizations Applied
- Reduced sequence length (64 tokens)
- Batch size = 1
- Gradient accumulation
- Single epoch training

## Usage
python scripts/train_cpu.py
python scripts/inference_cpu.py
