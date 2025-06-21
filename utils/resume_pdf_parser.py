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

from utils.llm import call_llm_with_timeout

logger = logging.getLogger("resume_pdf_parser")

load_dotenv(dotenv_path=".env.local")

api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)


async def parse_resume_pdf(file: UploadFile) -> Tuple[bool, str]:
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
            md_text = faback_to_markitdown(file_bytes=file_bytes, file=file)
            

        if not md_text or md_text.strip() == "":
            logger.error("❌ Parsed markdown is empty.")
            return False, "Error: Failed to extract content from PDF."

        # logger.info(f"Resume Markdown: {md_text}")
        logger.info(f"✅ Resume parsed to markdown successfully")

        return await call_llm(md_text)

    except Exception as e:
        logger.error(f"❌ Unexpected error while parsing PDF: {e}")
        return False, "❌ Error: An unexpected error occurred while parsing PDF."
    

def faback_to_markitdown(file_bytes: bytes, file: UploadFile) -> str:
    converter = MarkItDown()
    stream = BytesIO(file_bytes)
    stream.name = file.filename  # Optional, for format detection
    stream.seek(0)
    md_text = converter.convert(stream).text_content
    if not md_text or md_text.strip() == "":
      logger.error("MarkItDown also failed to parse resume to markdown!")
    return md_text
    

async def call_llm(resume_parsed: str) -> Tuple[bool, str]:
    try:
        prompt = f"""You are an expert resume analyzer. Your job is to analyze a resume in markdown format and extract the following information:

1. **Experience** — Include the total experience of the person in years.
2. **Skills** — Include not only explicitly listed skills but also infer them from work experience, certifications, and projects.
3. **Projects** — List notable projects the person has worked on, with brief descriptions. Include all projects worked within the various organizations that the person has worked at.
4. **Organizations** — List companies or institutions the person has worked at or studied in.
5. **Certifications and Extras** — Include certifications, patents, publications, and education details.

Also, rate each skill on a scale of 1 to 5, based on how strongly it is reflected in the resume.

Return the result in the following structured JSON format:

{{
  "experience": "10 years",
  "skills": [{{"name": "Kotlin", "rating": 5}}],
  "projects": [
    {{
      "title": "Multi Modal Listing (MML)",
      "description": "Led development of an AI-enhanced product listing feature in Android using server-based and on-device AI models. Reduced listing time and improved seller acquisition."
    }},
    {{
        "title": "Android App Modernization at VMware",
        "description": "Refactored large Java codebases to Kotlin, introduced Room DB, improved test coverage and CI/CD pipelines, and enforced architectural best practices."
    }}
  ],
  "organizations": ["Carousell", "VMWare"],
  "certifications_or_extras": {{
    "patents": [
      {{
        "title": "Reserving physical resources based upon a physical identifier",
        "status": "Issued",
        "patent_number": "US10547712"
      }}
    ],
    "education": "Bachelor of Engineering in Electronics and Communication, PESIT, VTU",
    "journals_or_publications": [
      "Navigation Drawer using Jetpack Compose",
      "Bottom Navigation and Navigation Drawer Using Scaffold from Jetpack Compose"
    ]
  }}
}}

Here is the resume:
{resume_parsed}
"""

        response = await call_llm_with_timeout(client, prompt)
        raw = response.strip()
        logger.info(f"LLM output: {raw}")
        return True, raw

    except Exception as e:
        logger.error(f"⚠️ llm failed to parse resume: {repr(e)}")
        return False, "error"
