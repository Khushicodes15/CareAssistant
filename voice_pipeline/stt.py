import sounddevice as sd
import numpy as np
import whisper
import tempfile
import soundfile as sf

model = whisper.load_model("base")

def listen(duration: int = 5, sample_rate: int = 16000) -> str:
    print("[STT] Listening...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    print("[STT] Processing...")

    audio_flat = audio.flatten()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio_flat, sample_rate)
        result = model.transcribe(f.name)

    return result["text"].strip()