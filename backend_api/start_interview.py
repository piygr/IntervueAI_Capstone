# File: backend_api/start_interview.py

from fastapi import APIRouter, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from livekit import api
import uuid
import os
from dotenv import load_dotenv
import logging
import sys
from utils.resume_pdf_parser import parse_resume_pdf
from utils.utils import update_session

logger = logging.getLogger("start-interview")

load_dotenv('.env.local')

router = APIRouter(prefix="/api")

LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_SERVER_URL = os.getenv("LIVEKIT_URL")

# print(LIVEKIT_API_KEY, LIVEKIT_SERVER_URL)

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

    # Generate LiveKit access token for candidate
    #token = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET, identity=interview_id)
    #token.add_grant(VideoGrant(room_join=True, room=room_name))
    #participant_token = token.to_jwt()

    # Log or store the resume, jobId, and room_name if needed
    print(f"Interview session created: room={room_name}, jobId={jobId}")
    print("interview_id: ", interviewId)

    session_dict = dict(room=room_name)
    update_session(interviewId, session_dict)

    return JSONResponse(content={
        "participantToken": token,
        "serverUrl": LIVEKIT_SERVER_URL,
        "roomName": room_name
    })