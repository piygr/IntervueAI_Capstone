import os
import logging
from dotenv import load_dotenv
from google import genai
import re
import json
from utils.session import get_google_api_key
from utils.llm import call_llm_with_timeout

logger = logging.getLogger(__name__)

load_dotenv(dotenv_path=".env.local")


#client = genai.Client(api_key=api_key)


class InterviewPlanner:
    def __init__(self, prompt_file_path="prompts/interview_planner.txt") -> None:
        api_key, api_key_index = get_google_api_key()
        self.api_key_index = api_key_index
        self.client = genai.Client(api_key=api_key)
        
        self.system_prompt = ""
        if os.path.exists(prompt_file_path):
            with open(prompt_file_path, 'r') as f:
                self.system_prompt = f.read()

    async def generate_interview_plan(self, jd_json, resume_json, scheduled_interview_duration_in_minutes) -> str:
        """
        Geerate interview plan as per jd & resume.
        """

        prompt = self.system_prompt.format(jd=jd_json, resume=resume_json, scheduled_duration_in_minutes=scheduled_interview_duration_in_minutes)
        interview_plan = await self.call_llm(prompt)
        if interview_plan:
            try:
                interview_plan_json = json.loads(interview_plan)
                return interview_plan_json
            except Exception as e:
                logger.debug(interview_plan)
                logger.error(f"Error parsing json interview plan: {e}")
                return interview_plan
                
        return {}


    async def call_llm(self, prompt: str) -> str:
        try:
            response = await call_llm_with_timeout(self.client, prompt)
            
            logger.info(f"LLM output: {response}")
            return response

        except Exception as e:
            err = str(e)
            logger.info(f"⚠️ llm failed to create intervue plan: {err}")

            return ""
