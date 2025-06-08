# File: main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend_api.start_interview import router as st_router
from backend_api.jd import router as jd_router

app = FastAPI()

# Allow frontend dev server access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js frontend port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include backend APIs
app.include_router(st_router)
app.include_router(jd_router)

@app.get("/")
def root():
    return {"message": "AI Interview Agent backend is running."}
