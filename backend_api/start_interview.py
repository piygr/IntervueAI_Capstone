# File: backend_api/start_interview.py

from fastapi import APIRouter, UploadFile, Form, Request
from fastapi.responses import JSONResponse, Response
from livekit import api
import uuid
import os
from dotenv import load_dotenv
import logging
import sys
from utils.resume_pdf_parser import parse_resume_pdf
from utils.interview_planner import generate_interview_plan
from utils.session import update_session, fetch_session

logger = logging.getLogger("start-interview")

load_dotenv('.env.local')

router = APIRouter(prefix="/api")

LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_SERVER_URL = os.getenv("LIVEKIT_URL")

print(LIVEKIT_API_KEY, LIVEKIT_SERVER_URL)

@router.post("/start-interview")
async def start_interview(interviewId: str = Form(...), jobId: str = Form(...)):
    logger.info(f"Starting interview for JD: {jobId}")
    room_name = f"interview-{interviewId}"

    token = api.AccessToken() \
    .with_identity(interviewId) \
    .with_name("Candidate") \
    .with_grants(api.VideoGrants(
        room_join=True,
        room=room_name,
    )).to_jwt()

    
    # Log or store the resume, jobId, and room_name if needed
    print(f"Interview session created: room={room_name}, jobId={jobId}")
    print("interview_id: ", interviewId)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    jd_path = os.path.join(script_dir, "..", "job_description", f"{jobId}.json")

    with open(jd_path, "r", encoding="utf-8") as file:
        jd_json = file.read()

    session_dict = fetch_session(interviewId)
    resume_json = session_dict.get('resume')
    
    if resume_json:
        interview_plan = await generate_interview_plan(jd_json, resume_json, 45)
        if interview_plan:
            session_dict = dict(room=room_name, 
                                JD=f"{jobId}", 
                                resume=resume_json, 
                                interview_plan=interview_plan)
            
            update_session(interviewId, session_dict)

            return JSONResponse(content={
                "participantToken": token,
                "serverUrl": LIVEKIT_SERVER_URL,
                "roomName": room_name
            })
        else:
            return JSONResponse(
                status_code=500,
                content="Error creatig interview plan"
            )
    else:
        return JSONResponse(
                status_code=500,
                content="Error parsing the resume"
            )