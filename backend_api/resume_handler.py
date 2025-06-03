from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import JSONResponse
import uuid
import subprocess
from livekit import api
import os
from dotenv import load_dotenv
load_dotenv()

# Environment variables or secure config
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")


router = APIRouter(prefix="/api")

@router.post("/start-interview")
async def start_interview(resume: UploadFile, jdId: str = Form(...)):
    # Mock storing resume and jdId
    interview_id = str(uuid.uuid4())
    token = api.AccessToken() \
    .with_identity("interviewer-bot") \
    .with_name("Interviewer Bot") \
    .with_grants(api.VideoGrants(
        room_join=True,
        room=f"interview-{interview_id}",
    )).to_jwt()
    # In production, you would store this metadata in a DB
    print(f"Received resume for JD: {jdId}, Interview ID: {interview_id}")


    # Save resume locally or to a DB here...

    # Start the LiveKit Agent process
    env = os.environ.copy()
    env["ROOM"] = f"interview-{interview_id}"

    subprocess.Popen([
        "/Users/piyushgrover/.pyenv/versions/interview-agent-env/bin/python",
        "agents/interview_agent.py",
        f"interview-{interview_id}"
    ], env=env)
    return JSONResponse(content={"interviewId": interview_id, "token": token})