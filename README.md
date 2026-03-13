# 🧓 CareAssistant — Offline AI Voice Assistant for Elderly & Visually Impaired

An end-to-end offline AI voice assistant designed for elderly and visually impaired users. Built with custom-trained ML models, a FastAPI backend, and a fully offline voice pipeline — no cloud required.

> **2nd Year Student Project** | Python · PyTorch · FastAPI · Whisper · Piper TTS · SpeechBrain

---

## 🎯 What It Does

- 🎙️ **Voice Login** — Log in by speaking. No typing required. Uses speaker embeddings (ECAPA-TDNN) and cosine similarity to verify identity.
- 🧠 **Intent Classification** — Custom fine-tuned DistilBERT model classifies 15 voice commands (set alarm, call contact, emergency, medicine reminder, etc.) with 98% accuracy.
- 💊 **Smart Medicine Reminders** — Random Forest model predicts the optimal reminder time based on the user's habits, missed doses, sleep schedule, and activity level.
- 📢 **Text-to-Speech** — Fully offline TTS using Piper (en_US-lessac-medium voice).
- 🎤 **Speech-to-Text** — Offline transcription using OpenAI Whisper (base model).
- 🔁 **Always-On Loop** — Wake word detection using Porcupine ("jarvis"), then listens and responds.
- 🌐 **Offline-First** — All core features work without internet. LLM fallback (Phi-3 via Ollama) handles open-ended questions when internet is available.

---

## 🏗️ Architecture

```
User speaks "Jarvis"
       ↓
Wake Word Detection (Porcupine)
       ↓
Speech-to-Text (Whisper.cpp)
       ↓
Intent Classifier (DistilBERT fine-tuned)
       ↓
Recognized? → FastAPI Backend → Response
Not recognized? → Phi-3 LLM (Ollama) → Response
       ↓
Text-to-Speech (Piper TTS)
       ↓
Spoken response to user
```

---

## 🤖 Models Trained

| Model | Type | Metric |
|-------|------|--------|
| Intent Classifier | DistilBERT fine-tune | 98% accuracy |
| Medication Adherence | Random Forest Regressor | R² = 0.82, MAE = 5.7 min |
| Voice Authentication | ECAPA-TDNN speaker embeddings | Cosine similarity ≥ 0.70 |

All models are trained from scratch (or fine-tuned) — not just API wrappers.

---

## 🗂️ Project Structure

```
care-assistant/
├── ml/
│   ├── intent_classifier/       # DistilBERT fine-tuning pipeline
│   │   ├── dataset/             # raw + augmented intent data
│   │   ├── train.py             # training script
│   │   ├── predict.py           # inference
│   │   └── evaluate.py          # metrics + confusion matrix
│   ├── medication_adherence/    # Random Forest reminder predictor
│   │   ├── dataset/             # synthetic data generator
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   └── voice_auth/              # Speaker verification
│       ├── embedding_model.py   # ECAPA-TDNN wrapper
│       ├── enroll.py            # voice enrollment
│       └── verify.py            # cosine similarity login
├── backend/                     # FastAPI REST API
│   ├── main.py
│   ├── routers/                 # auth, commands, medicine, reminders, alarms
│   ├── services/                # business logic
│   └── models.py                # SQLAlchemy DB models
├── voice_pipeline/              # Always-on voice loop
│   ├── main_loop.py
│   ├── stt.py                   # Whisper wrapper
│   ├── tts.py                   # Piper wrapper
│   ├── wake_word.py             # Porcupine integration
│   └── llm_handler.py           # Ollama/Phi-3 fallback
└── frontend/                    # Next.js dashboard
```

---

## ⚙️ Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/Khushicodes15/CareAssistant.git
cd CareAssistant
```

### 2. Install dependencies
```bash
pip install torch transformers scikit-learn pandas fastapi uvicorn sqlalchemy
pip install openai-whisper pvporcupine speechbrain torchaudio sounddevice soundfile
pip install python-jose passlib python-multipart
```

### 3. Train the ML models
```bash
# Intent Classifier
cd ml/intent_classifier/dataset
python augment.py
cd ..
python train.py

# Medication Adherence Model
cd ../medication_adherence/dataset
python generate_synthetic.py
cd ..
python train.py
```

### 4. Enroll a voice profile
```bash
cd ml/voice_auth
python enroll.py
```

### 5. Run the backend
```bash
cd backend
uvicorn main:app --reload
```
API docs available at: `http://localhost:8000/docs`

### 6. Run the voice pipeline
```bash
cd voice_pipeline
python main_loop.py
```

---

## 🔧 Requirements

- Python 3.11
- PyTorch (CPU is fine)
- Piper TTS — download from [rhasspy/piper releases](https://github.com/rhasspy/piper/releases)
- Porcupine access key — free at [picovoice.ai](https://console.picovoice.ai)
- Ollama (optional, for LLM fallback) — [ollama.ai](https://ollama.ai)

---

## 📊 Model Results

**Intent Classifier (DistilBERT)**
- Training accuracy: 98%
- 15 intent classes
- Dataset: 512 augmented samples

**Medication Adherence (Random Forest)**
- R² Score: 0.82
- MAE: 5.72 minutes
- Dataset: 8,000 synthetic samples

**Voice Authentication (ECAPA-TDNN)**
- Similarity threshold: 0.70
- 3 voice samples per enrollment
- Fully offline, no cloud API

---

## 💡 Supported Voice Commands

| Command | Intent |
|---------|--------|
| "Jarvis, what time is it?" | check_time |
| "Set alarm for 8am" | set_alarm |
| "Remind me to take medicine" | medicine_reminder |
| "Call my daughter" | call_contact |
| "I need help" | emergency |
| "Play some music" | play_music |
| "What's the weather?" | check_weather |
| "Open YouTube" | open_app |
| "Set a reminder" | set_reminder |
| "Hello Jarvis" | greeting |

---

## 🚧 Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Training | PyTorch, Scikit-learn, HuggingFace Transformers |
| Speaker Verification | SpeechBrain ECAPA-TDNN |
| Wake Word | Porcupine (Picovoice) |
| Speech-to-Text | OpenAI Whisper |
| Text-to-Speech | Piper TTS |
| LLM Fallback | Ollama + Phi-3 Mini |
| Backend | FastAPI + SQLite + SQLAlchemy |
| Auth | JWT tokens + Voice biometrics |
| Frontend | Next.js + Tailwind CSS |

---

## 👩‍💻 Author

**Khushi** — 2nd Year CS Student  
[GitHub](https://github.com/Khushicodes15)