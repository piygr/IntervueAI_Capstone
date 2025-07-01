import logging
from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import JSONResponse
import uuid
import subprocess
from livekit import api
import os
import json
from dotenv import load_dotenv

from utils.jd_resume_matcher import compare_jd_resume
from utils.resume_pdf_parser import parse_resume_pdf
from utils.session import update_session, load_config
load_dotenv()

# Environment variables or secure config
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("resume-handler")

config = load_config()
router = APIRouter(prefix="/api")

@router.post("/handle-resume")
async def start_interview(resume: UploadFile, jobId: str = Form(...)):
    logger.info(f"Received resume for JD: {jobId}")
    interview_id = str(uuid.uuid4())
    success, resume_text = await parse_resume_pdf(resume)

    if not success:
        return JSONResponse(content={
            "status": "failure",
            "message": f"Error parsing resume. Please try again. {resume_text}"
        })

    script_dir = os.path.dirname(os.path.abspath(__file__))
    jd_path = os.path.join(script_dir, "..", "job_description", f"{jobId}.json")

    with open(jd_path, "r", encoding="utf-8") as file:
        jd_text = file.read()

    jd_resume_match = await compare_jd_resume(jd_text, resume_text)

    if jd_resume_match == "error":
        return JSONResponse(content={
            "status": "failure",
            "message": "Error comparing JD and resume. Please try again."
        })
    
    # Parse the JSON response from compare_jd_resume
    try:
        match_data = json.loads(jd_resume_match)
        overall_score = match_data.get('overall_score', 0)
        logger.info(f"Overall score: {overall_score}")
        
        if overall_score >= config.get('interview', {}).get('min_resume_matching_score', 6):
            logger.info(f"Overall score is good: {overall_score}")
            # If score is good, return success with score and proceed to interview
            session_dict = dict(JD=f"{jobId}", resume=resume_text)
            update_session(interview_id, session_dict)
            return JSONResponse(content={
                "status": "success",
                "message": "Your profile matches well with the job requirements!",
                "interviewId": interview_id,
                "score": overall_score,
                "matchDetails": match_data
            })
        else:
            # If score is low, return failure with score and message
            return JSONResponse(content={
                "status": "failure",
                "message": "Your profile doesn't match the job requirements well enough. Please try applying for a different position.",
                "score": overall_score,
                "matchDetails": match_data
            })
            
    except json.JSONDecodeError:
        # Handle case where JSON parsing fails
        return JSONResponse(
            status_code=500,
            content={
                "status": "failure",
                "message": "Error processing resume. Please try again."
            }
        )