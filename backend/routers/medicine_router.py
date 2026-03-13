from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
import models
import schemas
import auth
import os
import importlib.util
from datetime import datetime, timedelta


_med_predict_path = os.path.join(os.path.dirname(__file__), "../../ml/medication_adherence/predict.py")
_spec = importlib.util.spec_from_file_location("med_predict", _med_predict_path)
_med_predict = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_med_predict)
predict_reminder_offset = _med_predict.predict_reminder_offset

router = APIRouter(prefix="/medicine", tags=["medicine"])

@router.post("/", response_model=schemas.MedicineOut)
def add_medicine(
    medicine: schemas.MedicineCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    new_medicine = models.Medicine(
        user_id=current_user.id,
        medicine_name=medicine.medicine_name,
        scheduled_time=medicine.scheduled_time
    )
    db.add(new_medicine)
    db.commit()
    db.refresh(new_medicine)
    return new_medicine

@router.get("/", response_model=list[schemas.MedicineOut])
def get_medicines(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    return db.query(models.Medicine).filter(
        models.Medicine.user_id == current_user.id,
        models.Medicine.active == True
    ).all()

@router.post("/log", response_model=schemas.MedicineLogOut)
def log_medicine(
    log: schemas.MedicineLogCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    medicine = db.query(models.Medicine).filter(
        models.Medicine.id == log.medicine_id,
        models.Medicine.user_id == current_user.id
    ).first()

    if not medicine:
        raise HTTPException(status_code=404, detail="Medicine not found")

    new_log = models.MedicineLog(
        user_id=current_user.id,
        medicine_id=log.medicine_id,
        taken=log.taken
    )
    db.add(new_log)
    db.commit()
    db.refresh(new_log)
    return new_log

@router.get("/reminder-time/{medicine_id}")
def get_smart_reminder_time(
    medicine_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    medicine = db.query(models.Medicine).filter(
        models.Medicine.id == medicine_id,
        models.Medicine.user_id == current_user.id
    ).first()

    if not medicine:
        raise HTTPException(status_code=404, detail="Medicine not found")

    week_ago = datetime.utcnow() - timedelta(days=7)
    logs = db.query(models.MedicineLog).filter(
        models.MedicineLog.medicine_id == medicine_id,
        models.MedicineLog.timestamp >= week_ago
    ).all()
    missed = sum(1 for log in logs if not log.taken)

    result = predict_reminder_offset(
        scheduled_time_str=medicine.scheduled_time,
        wake_time_str="07:00",
        sleep_time_str="23:00",
        activity_level="medium",
        missed_last_week=missed,
        day_of_week=datetime.now().strftime("%A"),
        taken_on_time=0
    )

    return result