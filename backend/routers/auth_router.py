from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database import get_db
import models
import schemas
import auth
import base64
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../ml/voice_auth"))
from verify import cosine_similarity

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup", response_model=schemas.UserOut)
def signup(user_data: schemas.UserCreate, db: Session = Depends(get_db)):
    existing = db.query(models.User).filter(
        models.User.username == user_data.username
    ).first()

    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")

    user = models.User(
        username=user_data.username.lower(),
        name=user_data.name
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@router.post("/voice-login", response_model=schemas.TokenOut)
def voice_login(data: schemas.VoiceLoginIn, db: Session = Depends(get_db)):
   
    audio_bytes = base64.b64decode(data.audio_base64)
    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

   
    users = db.query(models.User).filter(
        models.User.voice_embedding_path != None
    ).all()

    if not users:
        raise HTTPException(status_code=404, detail="No enrolled users found")

  
    best_user = None
    best_score = -1
    THRESHOLD = 0.70

    for user in users:
        if not os.path.exists(user.voice_embedding_path):
            continue
        stored_embedding = np.load(user.voice_embedding_path)

   
        sys.path.append(os.path.join(os.path.dirname(__file__), "../../ml/voice_auth"))
        from embedding_model import SpeakerEmbeddingModel
        model = SpeakerEmbeddingModel()
        live_embedding = model.get_embedding_from_array(audio_array)

        score = cosine_similarity(stored_embedding, live_embedding)
        if score > best_score:
            best_score = score
            best_user = user

    if best_user is None or best_score < THRESHOLD:
        raise HTTPException(
            status_code=401,
            detail="Voice not recognized"
        )

    token = auth.create_access_token({"sub": best_user.username})
    return {
        "access_token": token,
        "token_type": "bearer",
        "username": best_user.username,
        "name": best_user.name
    }


@router.get("/me", response_model=schemas.UserOut)
def get_me(current_user: models.User = Depends(auth.get_current_user)):
    return current_user