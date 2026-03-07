# Question Answering using BERT

This project implements a Question Answering system using the BERT (Bidirectional Encoder Representations from Transformers) architecture, using the efficient DistilBERT model for fast and accuracy results.

## Project Structure
- `data_loader_qa.py`: Downloads and prepares the SQuAD v1.1 dataset.
- `model_qa.py`: Configures the QA pipeline with a pre-finetuned DistilBERT model.
- `inference_qa.py`: A command-line script to interact with the model.
- `train_qa.py`: Demonstrates the logic for fine-tuning BERT on a subset of SQuAD dataset.

## High-Performance Lightweight Architecture
1. **DistilBERT**: Utilizes `distilbert-base-cased`, a smaller, faster version of BERT that retains 97% of its language understanding capabilities.
2. **Efficiency**: Optimized for fast inference on standard processors, making it suitable for a wide range of devices.
3. **Pre-trained Knowledge**: Uses state-of-the-art weights fine-tuned on the SQuAD dataset for immediate, high-accuracy results.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the interactive Question Answering tool:
   ```bash
   python inference_qa.py
   ```
3. (Optional) Run the training demonstration:
   ```bash
   python train_qa.py
   ```

## Expected Output
The system will take a paragraph (context) and a question, and produce a precise answer extraction along with a confidence score.

### Sample Interaction:
**Context**: "The Amazon rainforest... covers most of the Amazon basin of South America."  
**Question**: "Where is the Amazon rainforest located?"  
**Model Output**:  
> **Model's Answer**: South America  
> **Confidence Score**: 0.9854
