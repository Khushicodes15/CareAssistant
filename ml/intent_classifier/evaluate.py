import torch
import json
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt


SAVE_DIR = "saved_model"
DATA_PATH = "dataset/raw_intents.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 64

tokenizer = DistilBertTokenizer.from_pretrained(SAVE_DIR)
model = DistilBertForSequenceClassification.from_pretrained(SAVE_DIR)
model.to(DEVICE)
model.eval()

with open(f"{SAVE_DIR}/label_map.json", "r") as f:
    label_map = json.load(f)
reverse_label_map = {v: k for k, v in label_map.items()}

df = pd.read_csv(DATA_PATH)
le = LabelEncoder()
df["label"] = le.fit_transform(df["intent"])

_, X_test, _, y_test = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

all_preds = []
all_labels = y_test

for text in X_test:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        all_preds.append(pred)


intent_names = [reverse_label_map[i] for i in sorted(reverse_label_map.keys())]

print("\n── Classification Report ──\n")
print(classification_report(
    all_labels,
    all_preds,
    target_names=intent_names
))


cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=intent_names,
    yticklabels=intent_names,
    cmap="Blues"
)
plt.title("Intent Classifier - Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("\nConfusion matrix saved as confusion_matrix.png")