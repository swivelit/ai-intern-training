# 📝 GPT-2 Text Generation Project

This project focuses on fine-tuning a Large Language Model (LLM) to generate coherent, news-style text based on user-provided prompts.

## 🧠 Models Implemented
- **DistilGPT2**: A distilled, lightweight version of the GPT-2 transformer model. It provides a balance between performance and computational efficiency, making it ideal for localized fine-tuning.
- **Fine-tuning Strategy**: The model was fine-tuned on the **AG News Dataset**, specializing it in generating content related to World, Sports, Business, and Science/Technology.

## 🛠️ What We Use
- **PyTorch**: The deep learning framework used for model training and tensor operations.
- **HuggingFace Transformers**: Provides the pre-trained DistilGPT2 architecture and the `Trainer` API for efficient fine-tuning.
- **HuggingFace Datasets**: Used for seamless loading and preprocessing of the AG News corpus.
- **Pandas**: Used for initial data inspection and CSV management.

## 🎯 Output Results Detail
The project successfully fine-tuned the model to produce context-aware text. Below are examples of the typical output generated during testing:

- **Prompt**: "Technology is"
  - **Output**: *"Technology is developing an innovative technology to connect users to an online store."*
- **Prompt**: "Sports news:"
  - **Output**: *"Sports news: #2 player to return to U.S. Men's National Team"*
- **Prompt**: "The future of"
  - **Output**: *"The future of the Earth is still unclear, but the planet may have its first planetary satellite."*

The model demonstrates the ability to maintain topical consistency and valid sentence structure relevant to the training dataset.
