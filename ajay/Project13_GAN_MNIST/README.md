# Project 13 — GAN on MNIST (Handwritten Digit Generation)

This project trains a simple DCGAN-style GAN on the MNIST dataset to generate handwritten digits.

## Outputs
- Saves sample image grids during training to:
  - `outputs/samples/epoch_XXX.png`
  - `outputs/samples/final.png`
- Saves model weights to:
  - `outputs/weights/G_final.pt`, `outputs/weights/D_final.pt`

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
