import subprocess
import os
import sys

PIPER_PATH = os.path.join(os.path.dirname(__file__), "piper", "piper.exe")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "piper", "en_US-lessac-medium.onnx")

def speak(text: str):
    try:
        process = subprocess.Popen(
            [PIPER_PATH, "--model", MODEL_PATH, "--output_raw"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        raw_audio, _ = process.communicate(input=text.encode())

        import sounddevice as sd
        import numpy as np
        audio = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
        sd.play(audio, samplerate=22050)
        sd.wait()

    except Exception as e:
        print(f"[TTS ERROR] {e}")
        print(f"[TTS] {text}")