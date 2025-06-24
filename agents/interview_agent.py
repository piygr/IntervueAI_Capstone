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
    RoomInputOptions,
    RunContext,
    function_tool,
    get_job_context
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

You will be given a **job description** and a **candidate's resume** in JSON format. Use this information to prepare and ask **2 interview questions only**, 
starting with general or introductory topics and the next will be a very basic coding question on data structure and alogrithm.

Conduct the interview in an **interactive manner**:
- Ask one question at a time.
- Wait for the candidate's response before proceeding to the next.
- Do **not mention or reference** the job description or resume explicitly.
- Frame each question naturally, as a human interviewer would.

**Guidelines:**
- Tailor questions based on the candidate's past roles, projects, and skills.
- Align the focus of the questions with the job's requirements, but keep the language conversational.
- Ask a mix of behavioral, situational, and technical questions.
- When asking the coding question, frame the question in a manner that candidate can understand and also send the question text to code editor by calling the tool: 
"send_question_to_code_editor(question: str)
 Use this tool to send coding question or text to the code editor
Args:
    question: The question text to send to code editor."
- For coding question, keep some information ambiguous and let the cnadidate ask clarification question and then answer them just like a candidate would ask in a real interview. Do not send the clarification to code editor.
- When candidate submits the code, ask followup question on time complexity and space complexity if it's not already answered in the submitted code text.
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

    @function_tool()
    async def send_question_to_code_editor(self, context: RunContext, question: str):
        """Use this tool to send coding question or text to the code editor.
        Args:
            question: The question text to send to code editor.
        """
        logger.info(f"💬 coding question: {question}")
        room = get_job_context().room
        await room.local_participant.publish_data(
            question.encode("utf-8"),
            reliable=True,
            topic="code-editor"
        )
        await context.session.say("please open the code editor now.")


    async def handle_data_received(self, packet):
        # packet is a livekit.rtc.DataPacket
        data = packet.data
        topic = packet.topic
        participant = packet.participant.identity  # if you need it
        logger.info(f"💬 Received {len(data)} bytes on topic {topic}")
        if topic == "code-editor":
            code = data.decode("utf-8")

            marker = "// Type your code below this:"
            parts = code.split(marker, 1)
            if len(parts) > 1:
                code = parts[1].strip()
                if code:
                    final_marker = "<--final code-->"
                    if code.strip().endswith(final_marker):
                        code = code.rstrip().removesuffix(final_marker).rstrip()
                        if code:
                            logger.info(f"💬 Code: {code}")
                            # await self.session.say("Thank you for the submission.")
                            handle = self.session.generate_reply(user_input=code, instructions="User has submitted the code. Evaluate the code correctness and ask further question on it.")
                            await handle
                        else:
                            await self.session.say("Your submission is empty. Please submit full code")
                    else:
                        logger.info(f"💬 streaming code: {code}")
            else:
                await self.session.say("Please do not delete any existing code or text from the editor.")


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
        asyncio.create_task(agent.handle_data_received(packet))

    ctx.room.on("data_received", on_data_received)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    await session.start(room=ctx.room, agent=agent)

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )