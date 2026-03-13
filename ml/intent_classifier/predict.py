import os
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import json

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_model")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 64

# ── Load Model ───────────────────────────────────────────
tokenizer = DistilBertTokenizer.from_pretrained(SAVE_DIR)
model = DistilBertForSequenceClassification.from_pretrained(SAVE_DIR)
model.to(DEVICE)
model.eval()

# ── Load Label Map ───────────────────────────────────────
with open(f"{SAVE_DIR}/label_map.json", "r") as f:
    label_map = json.load(f)

# reverse: number → intent name
reverse_label_map = {v: k for k, v in label_map.items()}

# ── Predict Function ─────────────────────────────────────
def predict_intent(text: str) -> dict:
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
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        confidence, predicted = torch.max(probs, dim=1)

    intent = reverse_label_map[predicted.item()]
    confidence_score = confidence.item()

    return {
        "text": text,
        "intent": intent,
        "confidence": round(confidence_score, 4)
    }


if __name__ == "__main__":
    test_sentences = [
        "open youtube",
        "set alarm for 8am",
        "remind me to take medicine",
        "what time is it",
        "call my daughter",
        "i need help",
        "play some music",
        "what's the weather",
        "do i have internet",
        "thank you",
        "hello",
        "never mind",
        "what day is it today",
        "set a reminder for tomorrow",
        "play a video"
    ]

    print("\n── Intent Predictions ──\n")
    for sentence in test_sentences:
        result = predict_intent(sentence)
        print(f"Input    : {result['text']}")
        print(f"Intent   : {result['intent']}")
        print(f"Confidence: {result['confidence']}")
        print("─" * 40)