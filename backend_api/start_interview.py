# File: backend_api/start_interview.py

from fastapi import APIRouter, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from livekit import api
import uuid
import os
import subprocess
from dotenv import load_dotenv
import logging
import sys

logging.basicConfig(level=logging.INFO)
load_dotenv('.env.local')

router = APIRouter(prefix="/api")

LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_SERVER_URL = os.getenv("LIVEKIT_URL")

print(LIVEKIT_API_KEY, LIVEKIT_SERVER_URL)

@router.post("/start-interview")
async def start_interview(resume: UploadFile, jobId: str = Form(...)):
    # Generate a unique interview room name
    interview_id = str(uuid.uuid4())
    room_name = f"interview-{interview_id}"

    token = api.AccessToken() \
    .with_identity(interview_id) \
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

    env = os.environ.copy()
    #env["ROOM_NAME"] = room_name
    env["INTERVIEW_ID"] = interview_id
    env["RESUME_TEXT"] = ""
    env["JD"] = f"{jobId} data"

    #log_file = open(f"logs/agent_{room_name}.log", "w")
    subprocess.Popen([
        "/Users/piyushgrover/.pyenv/versions/interview-agent-env/bin/python",
        "agents/interview_agent.py",
        "dev",
        "--room",
        room_name
    ], env=env,
    stderr=sys.stderr,
    stdout=sys.stdout)

    return JSONResponse(content={
        "participantToken": token,
        "serverUrl": LIVEKIT_SERVER_URL,
        "roomName": room_name
    })