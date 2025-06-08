from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv
import logging
import json

logging.basicConfig(level=logging.INFO)
load_dotenv('.env.local')

router = APIRouter(prefix="/api")

@router.get("/jobs")
def fetch_job_by_id(jdId: str = Query(None)):
    jobs = []
    if jdId:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        jd_path = os.path.join(script_dir, "..", "job_description", f"{jdId}.json")
        with open(jd_path, 'r') as f:
            try:
                data = json.load(f)
                jobs.append(data)
            except json.JSONDecodeError:
                data = {}  # empty or corrupt file


    
    return JSONResponse(content={
        "jobs": jobs
    })