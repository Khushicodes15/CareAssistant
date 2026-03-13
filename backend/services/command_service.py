import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../ml/intent_classifier"))
from predict import predict_intent

def classify_command(text: str) -> dict:
    return predict_intent(text)