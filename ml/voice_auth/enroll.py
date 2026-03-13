import os
import json
import numpy as np
import sounddevice as sd
import soundfile as sf
from embedding_model import SpeakerEmbeddingModel

# ── Config ───────────────────────────────────────────────
EMBEDDINGS_DIR = "embeddings"
SAMPLE_RATE = 16000
RECORD_SECONDS = 5
NUM_SAMPLES = 3  # record 3 voice samples per user

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# ── Load Model ───────────────────────────────────────────
model = SpeakerEmbeddingModel()

# ── Record Audio ─────────────────────────────────────────
def record_audio(duration: int = RECORD_SECONDS) -> np.ndarray:
    print(f"\nRecording for {duration} seconds... Speak now!")
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    print("Recording done.")
    return audio.flatten()

# ── Enroll User ──────────────────────────────────────────
def enroll_user(username: str):
    print(f"\n── Enrolling user: {username} ──")
    print(f"You will be recorded {NUM_SAMPLES} times.")
    print("Say your name or a passphrase each time.\n")

    embeddings = []

    for i in range(NUM_SAMPLES):
        print(f"Sample {i+1} of {NUM_SAMPLES}")
        input("Press Enter when ready...")

        audio = record_audio()

        # save temp audio
        temp_path = f"{EMBEDDINGS_DIR}/temp_{username}_{i}.wav"
        sf.write(temp_path, audio, SAMPLE_RATE)

        # get embedding
        embedding = model.get_embedding(temp_path)
        embeddings.append(embedding)

        # remove temp file
        os.remove(temp_path)

        print(f"Sample {i+1} captured.\n")

    # average all embeddings
    avg_embedding = np.mean(embeddings, axis=0)

    # save embedding
    embedding_path = f"{EMBEDDINGS_DIR}/{username}.npy"
    np.save(embedding_path, avg_embedding)

    # update user registry
    registry_path = f"{EMBEDDINGS_DIR}/registry.json"
    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            registry = json.load(f)
    else:
        registry = {}

    registry[username] = {
        "embedding_path": embedding_path,
        "enrolled": True
    }

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"✓ User '{username}' enrolled successfully.")
    print(f"  Embedding saved to {embedding_path}")

# ── Main ─────────────────────────────────────────────────
if __name__ == "__main__":
    username = input("Enter username to enroll: ").strip().lower()
    enroll_user(username)
    