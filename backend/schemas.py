from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class UserCreate(BaseModel):
    username: str
    name: str

class UserOut(BaseModel):
    id: int
    username: str
    name: str
    created_at: datetime

    class Config:
        from_attributes = True

class TokenOut(BaseModel):
    access_token: str
    token_type: str
    username: str
    name: str

class VoiceLoginIn(BaseModel):
    audio_base64: str  # base64 encoded audio


class MedicineCreate(BaseModel):
    medicine_name: str
    scheduled_time: str  # "08:00"

class MedicineOut(BaseModel):
    id: int
    medicine_name: str
    scheduled_time: str
    active: bool

    class Config:
        from_attributes = True


class MedicineLogCreate(BaseModel):
    medicine_id: int
    taken: bool

class MedicineLogOut(BaseModel):
    id: int
    medicine_id: int
    taken: bool
    timestamp: datetime

    class Config:
        from_attributes = True


class ReminderCreate(BaseModel):
    title: str
    time: str
    repeat: bool = False

class ReminderOut(BaseModel):
    id: int
    title: str
    time: str
    repeat: bool
    active: bool

    class Config:
        from_attributes = True


class AlarmCreate(BaseModel):
    label: str
    time: str

class AlarmOut(BaseModel):
    id: int
    label: str
    time: str
    active: bool

    class Config:
        from_attributes = True

class CommandIn(BaseModel):
    text: str
    user_id: int

class CommandOut(BaseModel):
    intent: str
    response: str
    confidence: float

class CommandHistoryOut(BaseModel):
    id: int
    command_text: str
    intent: str
    response: str
    timestamp: datetime

    class Config:
        from_attributes = True