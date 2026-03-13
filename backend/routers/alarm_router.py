from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
import models
import schemas
import auth

router = APIRouter(prefix="/alarms", tags=["alarms"])

@router.post("/", response_model=schemas.AlarmOut)
def create_alarm(
    alarm: schemas.AlarmCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    new_alarm = models.Alarm(
        user_id=current_user.id,
        label=alarm.label,
        time=alarm.time
    )
    db.add(new_alarm)
    db.commit()
    db.refresh(new_alarm)
    return new_alarm

@router.get("/", response_model=list[schemas.AlarmOut])
def get_alarms(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    return db.query(models.Alarm).filter(
        models.Alarm.user_id == current_user.id,
        models.Alarm.active == True
    ).all()

@router.delete("/{alarm_id}")
def delete_alarm(
    alarm_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    alarm = db.query(models.Alarm).filter(
        models.Alarm.id == alarm_id,
        models.Alarm.user_id == current_user.id
    ).first()

    if not alarm:
        raise HTTPException(status_code=404, detail="Alarm not found")

    alarm.active = False
    db.commit()
    return {"message": "Alarm deleted"}