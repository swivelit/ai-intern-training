# Project 6: Question Answering using BERT (SQuAD v1.1)

## Overview
This project fine-tunes a Transformer-based Question Answering model on the
SQuAD v1.1 dataset using DistilBERT. A command-line interface is provided
to run inference on custom questions and contexts.

## Dataset
- SQuAD v1.1 (Hugging Face `datasets`)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
