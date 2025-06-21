# Load environment variables from .env file
import logging
import asyncio
import os
from dotenv import load_dotenv
from google import genai
import re

logger = logging.getLogger("llm")

load_dotenv()

# Access your API key and initialize Gemini client correctly
# api_key = os.getenv("GEMINI_API_KEY")
# client = genai.Client(api_key=api_key)

async def call_llm_with_timeout(client, prompt, timeout=30):
    """Generate content using llm with a timeout"""
    logger.info("Starting LLM generation...")
    try:
        # Convert the synchronous generate_content call to run in a thread
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None, 
                lambda: client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
            ),
            timeout=timeout
        )
        logger.info("LLM generation completed")
        return extract_json_string(response.text)
    except TimeoutError:
        logger.error("LLM generation timed out!")
        raise
    except Exception as e:
        logger.error(f"Error in LLM generation: {str(e)}")
        raise


def extract_json_string(raw_output: str) -> str:
    # Remove leading/trailing code block markers like ```json or ```
    cleaned = re.sub(r"^```(?:json)?\s*|```$", "", raw_output.strip(), flags=re.MULTILINE)
    return cleaned.strip()
