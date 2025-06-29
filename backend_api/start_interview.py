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
from utils.session import update_session, fetch_session, load_config
import json

logger = logging.getLogger("start-interview")

load_dotenv('.env.local')

router = APIRouter(prefix="/api")

LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_SERVER_URL = os.getenv("LIVEKIT_URL")

@router.post("/start-interview")
async def start_interview(interviewId: str = Form(...), jobId: str = Form(...)):
    config = load_config()
    print(LIVEKIT_API_KEY, LIVEKIT_SERVER_URL)
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
    jd_json = {}
    with open(jd_path, "r", encoding="utf-8") as file:
        jd_text = file.read()
        if jd_text:
            jd_json = json.loads(jd_text)

    session_dict = fetch_session(interviewId)
    resume_json = session_dict.get('resume')
    
    if resume_json:
        interview_plan = await generate_interview_plan(jd_json, resume_json, config.get('interview', {}).get('duration_minutes'))
        if interview_plan:
            interview_context = dict(
                candidate_first_name=resume_json.get('candidate_first_name', ''),
                candidate_name=resume_json.get('candidate_name', ''),
                candidate_email=resume_json.get('candidate_email', ''),
                company_name=jd_json.get('company', ''),
                jd_role=jd_json.get('title', '')
            )
            interview_context = json.loads(json.dumps(interview_context))
            
            session_dict = dict(room=room_name, 
                                JD=f"{jobId}", 
                                resume=resume_json, 
                                interview_plan=interview_plan,
                                interview_context=interview_context)
            
            update_session(interviewId, session_dict)

            return JSONResponse(content={
                "participantToken": token,
                "serverUrl": LIVEKIT_SERVER_URL,
                "roomName": room_name,
                "interviewId": interviewId
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