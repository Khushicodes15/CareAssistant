from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
import models
import schemas
import auth
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../ml/intent_classifier"))
from predict import predict_intent

router = APIRouter(prefix="/command", tags=["command"])

def generate_response(intent: str, user: models.User, db: Session) -> str:
    if intent == "check_time":
        from datetime import datetime
        return f"The current time is {datetime.now().strftime('%I:%M %p')}"

    elif intent == "check_date":
        from datetime import datetime
        return f"Today is {datetime.now().strftime('%A, %B %d, %Y')}"

    elif intent == "check_weather":
        return "I need internet to check the weather. Please make sure you're connected."

    elif intent == "open_app":
        return "Opening YouTube for you."

    elif intent == "play_video":
        return "Playing a video on YouTube."

    elif intent == "play_music":
        return "Playing music for you."

    elif intent == "medicine_reminder":
        medicines = db.query(models.Medicine).filter(
            models.Medicine.user_id == user.id,
            models.Medicine.active == True
        ).all()
        if not medicines:
            return "You have no medicines scheduled."
        med_list = ", ".join([f"{m.medicine_name} at {m.scheduled_time}" for m in medicines])
        return f"Your medicines are: {med_list}"

    elif intent == "set_alarm":
        return "Please tell me the time for the alarm."

    elif intent == "set_reminder":
        return "What would you like me to remind you about?"

    elif intent == "call_contact":
        return "Who would you like to call?"

    elif intent == "emergency":
        return "Emergency detected! Alerting your emergency contacts now."

    elif intent == "check_internet":
        from internet_check import is_connected
        connected = is_connected()
        if connected:
            return "Yes, you have an active internet connection."
        return "You are currently offline."

    elif intent == "stop_action":
        return "Okay, stopping."

    elif intent == "greeting":
        return f"Hello {user.name}! How can I help you today?"

    elif intent == "thanks":
        return "You're welcome! Is there anything else I can help you with?"

    return "I'm not sure how to help with that. Could you say that again?"


@router.post("/", response_model=schemas.CommandOut)
def process_command(
    command: schemas.CommandIn,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    result = predict_intent(command.text)
    intent = result["intent"]
    confidence = result["confidence"]

    response = generate_response(intent, current_user, db)

    # save to history
    history = models.CommandHistory(
        user_id=current_user.id,
        command_text=command.text,
        intent=intent,
        response=response
    )
    db.add(history)
    db.commit()

    return {
        "intent": intent,
        "response": response,
        "confidence": confidence
    }


@router.get("/history", response_model=list[schemas.CommandHistoryOut])
def get_history(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    return db.query(models.CommandHistory).filter(
        models.CommandHistory.user_id == current_user.id
    ).order_by(models.CommandHistory.timestamp.desc()).limit(50).all()