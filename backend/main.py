from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import engine
import models
from routers import auth_router, command_router, medicine_router, reminder_router, alarm_router

# create tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Care Assistant API",
    description="Offline AI assistant for elderly care",
    version="1.0.0"
)

# cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# routers
app.include_router(auth_router.router)
app.include_router(command_router.router)
app.include_router(medicine_router.router)
app.include_router(reminder_router.router)
app.include_router(alarm_router.router)

@app.get("/")
def root():
    return {"message": "Care Assistant API is running"}

@app.get("/health")
def health():
    from internet_check import get_internet_status
    return {
        "status": "ok",
        "internet": get_internet_status()
    }

@app.post("/test-token")
def get_test_token(username: str):
    from auth import create_access_token
    token = create_access_token({"sub": username})
    return {"access_token": token}