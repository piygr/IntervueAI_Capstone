# File: main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend_api.start_interview import router as st_router
from backend_api.jd import router as jd_router
from backend_api.resume_handler import router as resume_handler_router
from dotenv import load_dotenv
from google import genai
import os

load_dotenv(dotenv_path=".env.local")

allowed_origins = os.getenv("ALLOWED_ORIGINS")
app = FastAPI()

# Allow frontend dev server access
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Next.js frontend port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include backend APIs
app.include_router(st_router)
app.include_router(jd_router)
app.include_router(resume_handler_router)

@app.get("/")
def root():
    return {"message": "AI Interview Agent backend is running."}
