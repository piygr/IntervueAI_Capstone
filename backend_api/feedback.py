from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv
import logging
import json
from utils.memory import MemoryManager
from agents.feedback_agent import FeedbackAgent
from utils.session import fetch_session

logging.basicConfig(level=logging.INFO)
load_dotenv('.env.local')

router = APIRouter(prefix="/api")

    
@router.get("/feedback/{interviewId}")
async def get_feedback(interviewId: str):
    try:
        session = fetch_session(interviewId)
        return session.get("feedback")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    