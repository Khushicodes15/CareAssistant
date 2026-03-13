import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3"

def ask_llm(prompt: str) -> str:
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }, timeout=30)

        if response.status_code == 200:
            return response.json().get("response", "").strip()
        return "I couldn't process that right now."

    except Exception as e:
        return "I'm having trouble thinking right now. Please try again."