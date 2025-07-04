from io import BytesIO
import os
from typing import Tuple
from fastapi import UploadFile
from markitdown import MarkItDown
import pymupdf4llm
import logging
import fitz
from dotenv import load_dotenv
from google import genai
import re, json
from utils.llm import call_llm_with_timeout
from utils.session import get_google_api_key

logger = logging.getLogger("resume_pdf_parser")

load_dotenv(dotenv_path=".env.local")

#api_key = get_google_api_key()
#client = genai.Client(api_key=api_key)

class ResumePDFParser:
    def __init__(self, prompt_file_path="prompts/resume_pdf_parser.txt") -> None:
        api_key, api_key_index = get_google_api_key()
        self.api_key_index = api_key_index
        
        self.client = genai.Client(api_key=api_key)
        
        self.system_prompt = ""
        if os.path.exists(prompt_file_path):
            with open(prompt_file_path, 'r') as f:
                self.system_prompt = f.read()

    async def parse_resume_pdf(self, file: UploadFile) -> Tuple[bool, str]:
        """
        Parse a PDF file and return the text content in markdown format.
        """

        try:
            file_bytes = await file.read()  # Read the file contents

            try:
                doc = fitz.open(stream=file_bytes, filetype="pdf")  # Try to open PDF
                md_text = pymupdf4llm.to_markdown(doc)
            except Exception as e:
                logger.error(f"❌ Failed to open PDF for pymupdf: {e}")

            if not md_text or md_text.strip() == "":
                logger.error("❌ pymupdf4llm failed to parse resume to markdown. Trying Markitdown.")
                md_text = self.faback_to_markitdown(file_bytes=file_bytes, file=file)
                

            if not md_text or md_text.strip() == "":
                logger.error("❌ Parsed markdown is empty.")
                return False, "Error: Failed to extract content from PDF."

            # logger.info(f"Resume Markdown: {md_text}")
            logger.info(f"✅ Resume parsed to markdown successfully")

            success, resume_parsed = await self.call_llm(md_text)
            if resume_parsed:
                text = resume_parsed.strip()
                cleaned = re.sub(r"```json|```", "", text).strip()
                resume_parsed_json = json.loads(cleaned)
                return success, resume_parsed_json
            else:
                logger.error("LLM failed to parse resume.")
                return success, None

        except Exception as e:
            logger.error(f"❌ Unexpected error while parsing PDF: {e}")
            return False, "❌ Error: An unexpected error occurred while parsing PDF."
        

    def faback_to_markitdown(self, file_bytes: bytes, file: UploadFile) -> str:
        converter = MarkItDown()
        stream = BytesIO(file_bytes)
        stream.name = file.filename  # Optional, for format detection
        stream.seek(0)
        md_text = converter.convert(stream).text_content
        if not md_text or md_text.strip() == "":
            logger.error("MarkItDown also failed to parse resume to markdown!")
            return md_text
        

    async def call_llm(self, resume_parsed: str) -> Tuple[bool, str]:
        try:
            prompt = self.system_prompt.format(resume_parsed=resume_parsed)

            response = await call_llm_with_timeout(self.client, prompt)
            raw = response.strip()
            logger.info(f"LLM output: {raw}")
            return True, raw

        except Exception as e:
            logger.error(f"⚠️ llm failed to parse resume: {repr(e)}")
            return False, "error"
