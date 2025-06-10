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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    jobs = []
    if jdId:
        jd_path = os.path.join(script_dir, "..", "job_description", f"{jdId}.json")
        with open(jd_path, 'r') as f:
            try:
                data = json.load(f)
                jobs.append(data)
            except json.JSONDecodeError:
                data = {}  # empty or corrupt file
    else:
        folder_path = os.path.join(script_dir, "..", "job_description")
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r') as file:
                    try:
                        data = json.load(file)
                        data['id'] = filename.replace('.json', '')
                        jobs.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON from {filename}: {e}")
    
        jobs = sorted(jobs, key=lambda item: item['id'])
    
    return JSONResponse(content={
        "jobs": jobs
    })