from sqlalchemy.orm import Session
import models
from datetime import datetime, timedelta

def get_medicines(user_id: int, db: Session):
    return db.query(models.Medicine).filter(
        models.Medicine.user_id == user_id,
        models.Medicine.active == True
    ).all()

def add_medicine(user_id: int, name: str, scheduled_time: str, db: Session):
    medicine = models.Medicine(
        user_id=user_id,
        medicine_name=name,
        scheduled_time=scheduled_time
    )
    db.add(medicine)
    db.commit()
    db.refresh(medicine)
    return medicine

def log_medicine_taken(user_id: int, medicine_id: int, taken: bool, db: Session):
    log = models.MedicineLog(
        user_id=user_id,
        medicine_id=medicine_id,
        taken=taken
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log

def get_missed_count_last_week(medicine_id: int, db: Session) -> int:
    week_ago = datetime.utcnow() - timedelta(days=7)
    logs = db.query(models.MedicineLog).filter(
        models.MedicineLog.medicine_id == medicine_id,
        models.MedicineLog.timestamp >= week_ago
    ).all()
    return sum(1 for log in logs if not log.taken)

def check_due_medicines(user_id: int, db: Session):
    now = datetime.now().strftime("%H:%M")
    return db.query(models.Medicine).filter(
        models.Medicine.user_id == user_id,
        models.Medicine.scheduled_time == now,
        models.Medicine.active == True
    ).all()