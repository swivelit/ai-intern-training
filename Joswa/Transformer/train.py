import requests
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from torch.optim import AdamW

# 📥 Download SQuAD dataset
url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
data = requests.get(url).json()

contexts = []
questions = []
answers = []

# 📚 Extract data
for article in data["data"]:
    for para in article["paragraphs"]:
        context = para["context"]
        for qa in para["qas"]:
            contexts.append(context)
            questions.append(qa["question"])
            answers.append(qa["answers"][0]["text"])

# ⚡ Reduce size (for CPU)
contexts = contexts[:200]
questions = questions[:200]
answers = answers[:200]

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# 🧠 Tokenize
encodings = tokenizer(
    questions,
    contexts,
    truncation=True,
    padding=True,
    return_tensors="pt"
)

# 🎯 Find correct start & end positions
start_positions = []
end_positions = []

for i in range(len(contexts)):
    answer = answers[i]
    context = contexts[i]

    start_idx = context.find(answer)

    if start_idx == -1:
        start_positions.append(0)
        end_positions.append(1)
    else:
        end_idx = start_idx + len(answer)

        tokenized = tokenizer(
            questions[i],
            context,
            truncation=True,
            padding=True,
            return_offsets_mapping=True
        )

        offsets = tokenized["offset_mapping"]

        start_token = 0
        end_token = 0

        for idx, (start, end) in enumerate(offsets):
            if start <= start_idx < end:
                start_token = idx
            if start < end_idx <= end:
                end_token = idx

        start_positions.append(start_token)
        end_positions.append(end_token)

start_positions = torch.tensor(start_positions)
end_positions = torch.tensor(end_positions)

# 📦 Dataset
dataset = TensorDataset(
    encodings["input_ids"],
    encodings["attention_mask"],
    start_positions,
    end_positions
)

loader = DataLoader(dataset, batch_size=4, shuffle=True)

optimizer = AdamW(model.parameters(), lr=5e-5)

# 🚀 Training loop
model.train()

for epoch in range(1):
    print("Epoch started...")

    for i, batch in enumerate(loader):
        input_ids, attention_mask, start_pos, end_pos = batch

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_pos,
            end_positions=end_pos
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 10 == 0:
            print(f"Step {i}, Loss: {loss.item()}")

    print("Epoch finished!")

# 💾 Save model
model.save_pretrained("qa_model")
tokenizer.save_pretrained("qa_model")

print("✅ Training completed successfully!")