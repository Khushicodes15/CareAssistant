import sys
import os
import requests
import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile

sys.path.append(os.path.dirname(__file__))

from stt import listen
from tts import speak
from internet_check import is_connected

BACKEND_URL = "http://localhost:8000"
TOKEN = None
CURRENT_USER = None
SAMPLE_RATE = 16000

def get_token_for_user(username: str) -> str:
    response = requests.post(f"{BACKEND_URL}/test-token?username={username}")
    return response.json()["access_token"]

def send_command(text: str, token: str) -> str:
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(
        f"{BACKEND_URL}/command/",
        json={"text": text, "user_id": 1},
        headers=headers
    )
    if response.status_code == 200:
        return response.json()["response"]
    return "Sorry, I couldn't process that."

def voice_login() -> bool:
    global TOKEN, CURRENT_USER

    speak("Please say your name or passphrase to log in.")

    audio = sd.rec(
        int(5 * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()

    audio_bytes = (audio.flatten() * 32768).astype(np.int16).tobytes()
    import base64
    audio_b64 = base64.b64encode(audio_bytes).decode()

    response = requests.post(
        f"{BACKEND_URL}/auth/voice-login",
        json={"audio_base64": audio_b64}
    )

    if response.status_code == 200:
        data = response.json()
        TOKEN = data["access_token"]
        CURRENT_USER = data["name"]
        speak(f"Welcome back {CURRENT_USER}!")
        return True
    else:
        speak("I didn't recognize your voice. Please try again.")
        return False

def main():
    global TOKEN, CURRENT_USER

    speak("Care assistant is starting up.")

    # try voice login
    logged_in = False
    for attempt in range(3):
        if voice_login():
            logged_in = True
            break

    if not logged_in:
        speak("Could not verify your identity. Using guest mode.")
        TOKEN = get_token_for_user("testuser")
        CURRENT_USER = "Guest"

    speak(f"Hello {CURRENT_USER}! Say 'hey assistant' or just speak a command.")

    while True:
        try:
            print("\n[MAIN] Waiting for command...")
            text = listen(duration=5)

            if not text:
                continue

            print(f"[MAIN] Heard: {text}")

            # check for exit
            if any(word in text.lower() for word in ["goodbye", "bye", "exit", "shutdown"]):
                speak("Goodbye! Take care.")
                break

            # send to backend
            response = send_command(text, TOKEN)
            print(f"[MAIN] Response: {response}")
            speak(response)

        except KeyboardInterrupt:
            speak("Shutting down. Goodbye!")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            speak("Something went wrong. Please try again.")

if __name__ == "__main__":
    main()