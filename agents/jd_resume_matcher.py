import os
import logging
from google import genai
from dotenv import load_dotenv

from utils.llm import call_llm_with_timeout

logger = logging.getLogger("resume_matcher")

load_dotenv(dotenv_path=".env.local")

api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

async def compare_jd_resume(job_description: str, resume: str) -> str:
    if job_description == "" or resume == "":
        return "error"
    
    try:
        prompt = f"""
You are an expert recruiter and candidate evaluator.

Your task is to assess how well a candidate's resume matches a given job description. Consider the following evaluation criteria:

1. **Skill Match** - Alignment of technical and soft skills with job requirements  
2. **Experience Relevance** - Quality and relevance of prior work experience to the role  
3. **Domain Knowledge** - Familiarity with the industry, tools, or problem space  
4. **Role Fit** - Appropriateness of the candidate's current/past roles for this job level and responsibility

For each criterion, assign a score from 1 to 10 (10 = excellent match). Then, compute an **overall score** (average of the four) as `overall_score`.

Return the result in the following structured JSON format:

{{
  "skill_match": 8,
  "experience_relevance": 7,
  "domain_knowledge": 6,
  "role_fit": 8,
  "overall_score": 7.25
}}

Do not include any explanation or commentary outside the JSON.

Here is the job description:
{job_description}

Here is the candidate's resume:
{resume}
"""

        response = await call_llm_with_timeout(client, prompt)
        raw = response.strip()
        logger.info(f"LLM output: {raw}")
        return raw

    except Exception as e:
        logger.error(f"⚠️ llm failed to parse resume: {str(e)}")
        return "error"