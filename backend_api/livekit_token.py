from fastapi import APIRouter, Form
from livekit import api
import os
from dotenv import load_dotenv
load_dotenv()

router = APIRouter(prefix="/api")

# Environment variables or secure config
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

#print("LIVEKIT_API_KEY", LIVEKIT_API_KEY)

@router.post("/get-token")
def get_token(interviewId: str = Form(...)):
    token = api.AccessToken() \
    .with_identity("python-bot") \
    .with_name("Python Bot") \
    .with_grants(api.VideoGrants(
        room_join=True,
        room=f"interview-{interviewId}",
    )).to_jwt()
    return {"token": token, "identity": interviewId}

# will automatically use the LIVEKIT_API_KEY and LIVEKIT_API_SECRET env vars

