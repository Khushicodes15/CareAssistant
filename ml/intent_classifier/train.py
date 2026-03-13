import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
import os


MODEL_NAME = "distilbert-base-uncased"
SAVE_DIR = "saved_model"
DATA_PATH = "dataset/augmented_intents.csv"
EPOCHS = 10
BATCH_SIZE = 16
MAX_LEN = 64
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")


df = pd.read_csv(DATA_PATH)
print(f"Dataset size: {len(df)} samples")
print(f"Intents found: {df['intent'].unique()}")

le = LabelEncoder()
df["label"] = le.fit_transform(df["intent"])
label_map = {k: int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
print(f"Label map: {label_map}")

os.makedirs(SAVE_DIR, exist_ok=True)
with open(f"{SAVE_DIR}/label_map.json", "w") as f:
    json.dump(label_map, f)
print("Label map saved.")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")


tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)


class IntentDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx]
        }


train_dataset = IntentDataset(X_train, y_train)
test_dataset = IntentDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

num_labels = len(label_map)
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
)
model.to(DEVICE)


optimizer = AdamW(model.parameters(), lr=LR)


def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


print("\nStarting training...")
best_accuracy = 0

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer)
    val_acc = evaluate(model, test_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_accuracy:
        best_accuracy = val_acc
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)
        print(f"  ✓ Model saved (best val accuracy: {best_accuracy:.4f})")

print(f"\nTraining complete. Best accuracy: {best_accuracy:.4f}")