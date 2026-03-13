from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
import models
import schemas
import auth

router = APIRouter(prefix="/reminders", tags=["reminders"])

@router.post("/", response_model=schemas.ReminderOut)
def create_reminder(
    reminder: schemas.ReminderCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    new_reminder = models.Reminder(
        user_id=current_user.id,
        title=reminder.title,
        time=reminder.time,
        repeat=reminder.repeat
    )
    db.add(new_reminder)
    db.commit()
    db.refresh(new_reminder)
    return new_reminder

@router.get("/", response_model=list[schemas.ReminderOut])
def get_reminders(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    return db.query(models.Reminder).filter(
        models.Reminder.user_id == current_user.id,
        models.Reminder.active == True
    ).all()

@router.delete("/{reminder_id}")
def delete_reminder(
    reminder_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    reminder = db.query(models.Reminder).filter(
        models.Reminder.id == reminder_id,
        models.Reminder.user_id == current_user.id
    ).first()

    if not reminder:
        raise HTTPException(status_code=404, detail="Reminder not found")

    reminder.active = False
    db.commit()
    return {"message": "Reminder deleted"}