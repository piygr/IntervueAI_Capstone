import os
import logging
import fitz
from dotenv import load_dotenv
from google import genai
import re, json
from utils.llm import call_llm_with_timeout
from utils.memory import MemoryManager
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path=".env.local")

api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

# === Interviewer Agent ===
class Interviewer:
    def __init__(self, prompt_file_path="prompts/interviewer.txt") -> None:
        self.prompt_file_path = prompt_file_path
        
        if os.path.exists(prompt_file_path):
            with open(prompt_file_path, 'r') as f:
                self.system_prompt = f.read()
        else:
            self.system_prompt = ""

        #logger.info(self.system_prompt)

    '''def decide_next_action(self, memory) -> str:
        # Placeholder logic - use LLM or rules
        if "don't know" in user_text.lower():
            return "hint"
        elif len(user_text.split()) < 10:
            return "probe"
        elif "done" in user_text.lower():
            return "end"
        return "next"'''
    
    async def decide_next_action(self, memory, interview_phase="ONGOING") -> str:
        
        action_json = await self.call_llm(memory=memory, interview_phase=interview_phase)
        
        return action_json


    async def call_llm(self, memory, interview_phase) -> str:
        try:
            prompt = self.system_prompt.format(memory=memory, interview_phase=interview_phase)
            #logger.info(prompt)
            response = await call_llm_with_timeout(client, prompt)
            #raw = response.text.strip()
            #cleaned = re.sub(r"```json|```", "", raw).strip()
            response_json = json.loads(response)
            logger.info(f"LLM output: {response_json}")
            return response_json

        except Exception as e:
            logger.info(f"⚠️ llm failed to parse resume: {e}")
            return ""
