import logging

from dotenv import load_dotenv
import asyncio
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    metrics,
    RoomInputOptions
)
from livekit.agents.llm.chat_context import ChatContext, ChatMessage
from livekit.plugins import (
    openai,
    silero,
    aws,
    google
)
#from livekit.plugins.turn_detector.english import EnglishModel

from utils.utils import fetch_session

import os

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")

llm = google.LLM(
        model="gemini-2.0-flash",
    )
stt = aws.STT()
tts = aws.TTS(
    voice="Kajal",
    language="en-IN",
    speech_engine="generative",
    api_key=os.getenv("AWS_ACCESS_KEY_ID"),
    api_secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region=os.getenv("AWS_REGION")
)


class Assistant(Agent):
    def __init__(self, job_description: str, resume: str) -> None:
        # This project is configured to use Deepgram STT, OpenAI LLM and Cartesia TTS plugins
        # Other great providers exist like Cerebras, ElevenLabs, Groq, Play.ht, Rime, and more
        # Learn more and pick the best one for your app:
        # https://docs.livekit.io/agents/plugins
        instructions = f"""
You are an AI interview agent conducting a structured job interview.

You will be given a **job description** and a **candidate's resume** in JSON format. Use this information to prepare and ask **10 interview questions**, starting with general or introductory topics and gradually progressing to more technical or role-specific areas.

Conduct the interview in an **interactive manner**:
- Ask one question at a time.
- Wait for the candidate's response before proceeding to the next.
- Do **not mention or reference** the job description or resume explicitly.
- Frame each question naturally, as a human interviewer would.

**Guidelines:**
- Tailor questions based on the candidate's past roles, projects, and skills.
- Align the focus of the questions with the job's requirements, but keep the language conversational.
- Ask a mix of behavioral, situational, and technical questions.
- Keep the tone professional and engaging.
- Do not answer the questions yourself.

Here is the job description:
{job_description}

Here is the candidate's resume:
{resume}
"""

        super().__init__(
            instructions=instructions,
            stt=stt,
            tts=tts,
            #turn_detection=EnglishModel,
            llm=llm
        )

    async def on_enter(self):
        # The agent should be polite and greet the user when it joins :)
        self.session.say("Hey, how are you doing today?")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    await ctx.connect(
        auto_subscribe=AutoSubscribe.AUDIO_ONLY,
    )
    logger.info(f"connecting to room {ctx.room.name}")
    logger.info(f"Interview ID: {os.getenv('INTERVIEW_ID')}")

    session_id = ctx.room.name.replace('interview-', '')
    session_details = fetch_session(session_id)
    jd_id = session_details.get("JD")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    jd_path = os.path.join(script_dir, "..", "job_description", f"{jd_id}.json")
    with open(jd_path, "r", encoding="utf-8") as file:
        jd_text = file.read()

    agent = Assistant(job_description=jd_text, resume=session_details.get("resume"))

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        min_endpointing_delay=0.5,
        max_endpointing_delay=5.0
    )

    def on_data_received(packet):
        asyncio.create_task(handle_data_received(packet))

    async def handle_data_received(packet):
        # packet is a livekit.rtc.DataPacket
        data = packet.data
        topic = packet.topic
        participant = packet.participant.identity  # if you need it
        logger.info(f"Received {len(data)} bytes on topic {topic}")
        if topic == "code-editor":
            msg = data.decode("utf-8")
            logger.info(f"Message: {msg}")
            # You can process/store the code as needed

    ctx.room.on("data_received", on_data_received)

    # --- Example: Send code to UI after participant joins ---
    async def send_code_to_ui():
        await asyncio.sleep(5)  # Wait for UI to be ready (optional)
        code_snippet = "// Here is a code snippet from the agent"
        await ctx.room.local_participant.publish_data(
            code_snippet.encode("utf-8"),
            reliable=True,
            topic="code-editor"
        )

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    # Start the session and send code after join
    await session.start(room=ctx.room, agent=agent)
    await send_code_to_ui()

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )