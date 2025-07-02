import os
import logging
from google import genai
from dotenv import load_dotenv
from utils.session import get_google_api_key
from utils.llm import call_llm_with_timeout

logger = logging.getLogger("resume_matcher")

load_dotenv(dotenv_path=".env.local")

#client = genai.Client(api_key=api_key)

class JDResumeMatcher:
    def __init__(self, prompt_file_path="prompts/jd_resume_matcher.txt") -> None:
        api_key, api_key_index = get_google_api_key()
        self.api_key_index = api_key_index
        
        self.client = genai.Client(api_key=api_key)
        
        self.system_prompt = ""
        if os.path.exists(prompt_file_path):
            with open(prompt_file_path, 'r') as f:
                self.system_prompt = f.read()

    async def compare_jd_resume(self, job_description: str, resume: str) -> str:
        if job_description == "" or resume == "":
            return "error"
        
        try:
            prompt = self.system_prompt.format(job_description=job_description, resume=resume)

            response = await call_llm_with_timeout(self.client, prompt)
            raw = response.strip()
            logger.info(f"LLM output: {raw}")
            return raw

        except Exception as e:
            logger.error(f"⚠️ llm failed to parse resume: {str(e)}")
            return "error"