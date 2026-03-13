import os
import json
import numpy as np
import sounddevice as sd
import soundfile as sf
from numpy.linalg import norm
from embedding_model import SpeakerEmbeddingModel

# ── Config ───────────────────────────────────────────────
EMBEDDINGS_DIR = "embeddings"
SAMPLE_RATE = 16000
RECORD_SECONDS = 5
SIMILARITY_THRESHOLD = 0.70


model = SpeakerEmbeddingModel()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (norm(a) * norm(b)))


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


def verify_user(username: str) -> dict:
    registry_path = f"{EMBEDDINGS_DIR}/registry.json"

   
    if not os.path.exists(registry_path):
        return {"authenticated": False, "reason": "No users enrolled"}

    with open(registry_path, "r") as f:
        registry = json.load(f)

    if username not in registry:
        return {"authenticated": False, "reason": f"User '{username}' not found"}

    
    embedding_path = registry[username]["embedding_path"]
    stored_embedding = np.load(embedding_path)


    print(f"\nVerifying user: {username}")
    print("Please say your name or passphrase.")
    input("Press Enter when ready...")
    audio = record_audio()

    
    temp_path = f"{EMBEDDINGS_DIR}/temp_verify.wav"
    sf.write(temp_path, audio, SAMPLE_RATE)

    live_embedding = model.get_embedding(temp_path)
    os.remove(temp_path)

  
    similarity = cosine_similarity(stored_embedding, live_embedding)
    authenticated = similarity >= SIMILARITY_THRESHOLD

    print(f"\nSimilarity score: {similarity:.4f}")
    print(f"Threshold       : {SIMILARITY_THRESHOLD}")
    print(f"Result          : {'✓ Authenticated' if authenticated else '✗ Failed'}")

    return {
        "authenticated": authenticated,
        "username": username,
        "similarity": round(similarity, 4),
        "reason": "Voice matched" if authenticated else "Voice did not match"
    }

def identify_user() -> dict:
    """
    Tries to identify who is speaking
    without knowing the username beforehand
    """
    registry_path = f"{EMBEDDINGS_DIR}/registry.json"

    if not os.path.exists(registry_path):
        return {"authenticated": False, "reason": "No users enrolled"}

    with open(registry_path, "r") as f:
        registry = json.load(f)

    if not registry:
        return {"authenticated": False, "reason": "No users enrolled"}

    print("\nIdentifying speaker...")
    print("Please say your name or passphrase.")
    input("Press Enter when ready...")
    audio = record_audio()

    temp_path = f"{EMBEDDINGS_DIR}/temp_identify.wav"
    sf.write(temp_path, audio, SAMPLE_RATE)
    live_embedding = model.get_embedding(temp_path)
    os.remove(temp_path)


    best_match = None
    best_score = -1

    for username in registry:
        stored = np.load(registry[username]["embedding_path"])
        score = cosine_similarity(stored, live_embedding)
        if score > best_score:
            best_score = score
            best_match = username

    authenticated = best_score >= SIMILARITY_THRESHOLD

    print(f"\nBest match      : {best_match}")
    print(f"Similarity score: {best_score:.4f}")
    print(f"Result          : {'✓ Authenticated' if authenticated else '✗ Failed'}")

    return {
        "authenticated": authenticated,
        "username": best_match if authenticated else None,
        "similarity": round(best_score, 4),
        "reason": f"Identified as {best_match}" if authenticated else "Could not identify speaker"
    }


if __name__ == "__main__":
    print("1. Verify specific user")
    print("2. Identify speaker")
    choice = input("Choose (1/2): ").strip()

    if choice == "1":
        username = input("Enter username: ").strip().lower()
        result = verify_user(username)
    else:
        result = identify_user()

    print(f"\nFinal result: {result}")