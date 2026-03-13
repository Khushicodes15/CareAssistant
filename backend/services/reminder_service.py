from sqlalchemy.orm import Session
import models
from datetime import datetime

def get_active_reminders(user_id: int, db: Session):
    return db.query(models.Reminder).filter(
        models.Reminder.user_id == user_id,
        models.Reminder.active == True
    ).all()

def create_reminder(user_id: int, title: str, time: str, repeat: bool, db: Session):
    reminder = models.Reminder(
        user_id=user_id,
        title=title,
        time=time,
        repeat=repeat
    )
    db.add(reminder)
    db.commit()
    db.refresh(reminder)
    return reminder

def deactivate_reminder(reminder_id: int, user_id: int, db: Session):
    reminder = db.query(models.Reminder).filter(
        models.Reminder.id == reminder_id,
        models.Reminder.user_id == user_id
    ).first()
    if reminder:
        reminder.active = False
        db.commit()
    return reminder

def check_due_reminders(user_id: int, db: Session):
    now = datetime.now().strftime("%H:%M")
    return db.query(models.Reminder).filter(
        models.Reminder.user_id == user_id,
        models.Reminder.time == now,
        models.Reminder.active == True
    ).all()