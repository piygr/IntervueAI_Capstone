# === Scoring Agent ===
import logging
from dotenv import load_dotenv

import json
import os
from utils.llm import call_llm_with_timeout
from utils.memory import MemoryManager
from utils.session import fetch_session, update_session
from google import genai

logger = logging.getLogger(__name__)
load_dotenv(dotenv_path=".env.local")

api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)


class FeedbackAgent:
    @staticmethod
    async def generate_feedback(memory: MemoryManager):
        # logger.debug(f'📖 Full interview conversation: {transcript}')
        transcript = memory.get_conversation()
        session_id = memory.session_id
        feedback_context = FeedbackAgent._get_feedback_context(session_id)

        prompt_file_path = "prompts/feedback.txt"

        if os.path.exists(prompt_file_path):
            with open(prompt_file_path, 'r') as f:
                system_prompt = f.read()

        feedback_prompt = system_prompt.format(transcript=transcript, interview_plan=memory.interview_plan.model_dump_json(), **feedback_context)
        # logger.debug(f'🎞️ feedback prompt: {feedback_prompt}')
        feedback = await call_llm_with_timeout(client=client, prompt=feedback_prompt)
        logger.debug(f'🎞️ Feedback generated. {feedback}')
        session_dict = dict(feedback=json.loads(feedback))
        update_session(session_id, session_dict)
    
    @staticmethod
    def _get_jd_json(jobId) -> dict :
        script_dir = os.path.dirname(os.path.abspath(__file__))
        jd_path = os.path.join(script_dir, "..", "job_description", f"{jobId}.json")

        with open(jd_path, "r", encoding="utf-8") as file:
            jd_text = file.read()
        return json.loads(jd_text)
    
    @staticmethod
    def _get_feedback_context(session_id,) -> dict:
        session_details = fetch_session(session_id)
        jd_json = FeedbackAgent._get_jd_json(session_details.get("JD"))

        return dict(
            role=jd_json.get("title"),
            jd_role=jd_json.get("title"),
            job_description=json.dumps(jd_json),
            resume=session_details.get("resume")
        )