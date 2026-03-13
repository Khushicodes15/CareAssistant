from sqlalchemy.orm import Session
import models
from datetime import datetime

def get_active_alarms(user_id: int, db: Session):
    return db.query(models.Alarm).filter(
        models.Alarm.user_id == user_id,
        models.Alarm.active == True
    ).all()

def create_alarm(user_id: int, label: str, time: str, db: Session):
    alarm = models.Alarm(
        user_id=user_id,
        label=label,
        time=time
    )
    db.add(alarm)
    db.commit()
    db.refresh(alarm)
    return alarm

def deactivate_alarm(alarm_id: int, user_id: int, db: Session):
    alarm = db.query(models.Alarm).filter(
        models.Alarm.id == alarm_id,
        models.Alarm.user_id == user_id
    ).first()
    if alarm:
        alarm.active = False
        db.commit()
    return alarm

def check_due_alarms(user_id: int, db: Session):
    now = datetime.now().strftime("%H:%M")
    return db.query(models.Alarm).filter(
        models.Alarm.user_id == user_id,
        models.Alarm.time == now,
        models.Alarm.active == True
    ).all()