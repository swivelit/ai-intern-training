Text Generation using GPT-2 (Small)

Project Overview

This project demonstrates how to fine-tune a pre-trained GPT-2 (Small) language model on a custom text corpus and generate meaningful text from a given prompt.

The implementation is done using Hugging Face Transformers, PyTorch, and a single-cell Python script, making it simple, efficient, and suitable for beginners, internships, and interviews.

Project Objectives

    Fine-tune GPT-2 (Small) on a custom dataset
    Understand causal language modeling
    Generate text based on a given prompt
    Learn Hugging Face Trainer workflow
    Save and reuse trained models

Dataset Description
    A small custom text corpus is used for training the model.

Training Data (train.txt) 
    Artificial Intelligence is transforming the world.
Machine learning allows systems to learn from data.
Deep learning uses neural networks with many layers.
Natural language processing helps computers understand text.
AI is used in healthcare, finance, and education.

Note:
The dataset is intentionally small to ensure fast training and easy understanding.
For better text quality, a larger dataset such as Wikipedia or news articles can be used.

Technologies Used
    Python 3.10
    PyTorch
    Hugging Face Transformers
    Hugging Face Datasets
    GPT-2 Small (117M parameters)

Installation Steps
    Create a virtual environment (recommended)
        python -m venv gpt2_env
        gpt2_env\Scripts\activate

    Install required libraries
        pip install torch transformers datasets

Project Workflow
Step 1: Dataset Creation
    A text file is created that contains the training corpus used for fine-tuning GPT-2.
Step 2: Model and Tokenizer Loading
    Loads the pre-trained GPT-2 Small model
    Sets the padding token to the EOS token
Step 3: Tokenization
    Converts text into token IDs
    Adds labels = input_ids
    This step is required for GPT-2 to compute training loss
Step 4: Model Fine-Tuning
    Uses Hugging Face Trainer
    Trains the model for 3 epochs
    Training loss decreases over time, confirming learning
Step 5: Text Generation
    Generates text from a given prompt
    Uses sampling-based generation (do_sample=True)
    Controlled using temperature, top_k, and top_p

Training Output
    Example training loss values:
        loss: 9.13 → 6.78 → 5.31 → 1.98 → 1.40 → 0.44
        Decreasing loss indicates successful fine-tuning.

Sample Generated Text
    Artificial Intelligence is transforming the world and is used in healthcare, finance, and education. Machine learning allows systems to learn from data and improve over time.

    Output quality depends on dataset size and training duration.

Output Files
    After training, the following directory is created automatically:

    gpt2_finetuned/
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer.json
    ├── vocab.json
    └── merges.txt

    These files can be reused for inference or deployment.

