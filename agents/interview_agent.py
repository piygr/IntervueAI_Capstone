import logging

from dotenv import load_dotenv
import asyncio

from livekit import rtc
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
    RoomOutputOptions,
    RoomIO,
    StopResponse,
    llm,
    utils
)

from livekit.agents.llm.chat_context import ChatContext, ChatMessage
from livekit.plugins import (
    openai,
    silero,
    aws,
    google,
    noise_cancellation
)
#from livekit.plugins.turn_detector.english import EnglishModel

from utils.session import fetch_session
from agents.coordinator import Coordinator, Memory
import os

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")

llm_model = google.LLM(
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
            llm=llm_model
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
    
    
    session_id = ctx.room.name.replace('interview-', '')

    ### Fetching session details eg. JD, Resume ####
    session_details = fetch_session(session_id)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    usage_collector = metrics.UsageCollector()

    # Log metrics and collect usage data
    def on_metrics_collected(agent_metrics: metrics.AgentMetrics):
        metrics.log_metrics(agent_metrics)
        usage_collector.collect(agent_metrics)

    agent = Coordinator(
        participant_identity=participant.identity,
        interview_plan=session_details.get("interview_plan"),
        interview_context=session_details.get("interview_context")
    )

    def on_data_received(packet):
        asyncio.create_task(agent.handle_data_received(packet))

    ctx.room.on("data_received", on_data_received)
    
    session = AgentSession[Memory](
        vad=ctx.proc.userdata["vad"],
        # minimum delay for endpointing, used when turn detector believes the user is done with their turn
        min_endpointing_delay=1,
        # maximum delay for endpointing, used when turn detector does not believe the user is done with their turn
        max_endpointing_delay=5.0,
        allow_interruptions=True,
        min_interruption_words=5,
        userdata=Memory(),
    )

    # Trigger the on_metrics_collected function when metrics are collected
    session.on("metrics_collected", on_metrics_collected)

    try:
        logger.info("Waiting for participant to join (max 30s)...")
        #participant = await asyncio.wait_for(ctx.wait_for_participant(), timeout=300)
        logger.info(f"Participant joined: {participant.identity}")
    except asyncio.TimeoutError:
        logger.warning("No participant joined within 30 seconds. Shutting down agent.")
        await ctx.close()
        return  # Exits the entrypoint, safely ends the subprocess

    await session.start(
        room=ctx.room, 
        agent=agent,
        room_input_options=RoomInputOptions(
            text_enabled=False,
            #noise_cancellation=noise_cancellation.BVC()
        ),
        room_output_options=RoomOutputOptions(
            transcription_enabled=True,
            audio_enabled=True,
        )
    )

'''

class Transcriber(Agent):
    def __init__(self, *, participant_identity: str):
        super().__init__(
            instructions="not-needed",
            stt=stt,
            tts=tts
        )
        self.participant_identity = participant_identity

    async def on_user_turn_completed(self, chat_ctx: llm.ChatContext, new_message: llm.ChatMessage):
        user_transcript = new_message.text_content
        logger.info(f"{self.participant_identity} -> {user_transcript}")
        self.session.say("I am listening...")
        raise StopResponse()
    
class SingleUserTranscriber:
    def __init__(self, ctx: JobContext):
        self.ctx = ctx
        self._session: AgentSession = None
        self._task: asyncio.Task = None

    def start(self):
        self.ctx.room.on("participant_connected", self.on_participant_connected)
        self.ctx.room.on("participant_disconnected", self.on_participant_disconnected)

    async def aclose(self):
        await utils.aio.cancel_and_wait(*self._task)

        await asyncio.gather(*[self._close_session(self._session)])

        self.ctx.room.off("participant_connected", self.on_participant_connected)
        self.ctx.room.off("participant_disconnected", self.on_participant_disconnected)

    def on_participant_connected(self, participant: rtc.RemoteParticipant):
        if self._session is not None:
            return

        logger.info(f"starting session for {participant.identity}")
        task = asyncio.create_task(self._start_session(participant))
        self._task = task

        def on_task_done(task: asyncio.Task):
            try:
                self._session = task.result()
            finally:
                self._task = None

        task.add_done_callback(on_task_done)

    def on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        if self._session is None:
            return

        logger.info(f"closing session for {participant.identity}")
        task = asyncio.create_task(self._close_session(self._session))
        self._task = task
        task.add_done_callback(lambda _: setattr(self, '_task', None))

    async def _start_session(self, participant: rtc.RemoteParticipant) -> AgentSession:
        if self._session is not None:
            return self._session

        session = AgentSession(
            vad=self.ctx.proc.userdata["vad"],
            tts=tts
        )
        room_io = RoomIO(
            agent_session=session,
            room=self.ctx.room,
            participant=participant,
            input_options=RoomInputOptions(
                # text input is not supported for multiple room participants
                # if needed, register the text stream handler by yourself
                # and route the text to different sessions based on the participant identity
                text_enabled=False,
            ),
            output_options=RoomOutputOptions(
                transcription_enabled=True,
                audio_enabled=True,
            ),
        )
        await room_io.start()
        await session.start(
            agent=Transcriber(
                participant_identity=participant.identity
            )
        )
        return session

    async def _close_session(self, sess: AgentSession) -> None:
        await sess.drain()
        await sess.aclose()


async def entrypoint2(ctx: JobContext):
    transcriber = SingleUserTranscriber(ctx)
    transcriber.start()

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    for participant in ctx.room.remote_participants.values():
        # handle all existing participants
        transcriber.on_participant_connected(participant)

    async def cleanup():
        await transcriber.aclose()

    ctx.add_shutdown_callback(cleanup)

'''


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )