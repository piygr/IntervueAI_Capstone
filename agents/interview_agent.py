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
)
from livekit.agents.llm.chat_context import ChatContext, ChatMessage
from livekit.plugins import (
    openai,
    silero,
    aws,
    google
)
#from livekit.plugins.turn_detector.english import EnglishModel


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
    def __init__(self) -> None:
        # This project is configured to use Deepgram STT, OpenAI LLM and Cartesia TTS plugins
        # Other great providers exist like Cerebras, ElevenLabs, Groq, Play.ht, Rime, and more
        # Learn more and pick the best one for your app:
        # https://docs.livekit.io/agents/plugins
        super().__init__(
            instructions="You are an interviewer agent.",
            stt=stt,
            tts=tts,
            #turn_detection=EnglishModel,
            #llm=llm
        )

    async def on_enter(self):
        # The agent should be polite and greet the user when it joins :)
        self.session.say("Hey, how are you doing today?")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    usage_collector = metrics.UsageCollector()

    #def on_transcription_received(segments):
    #    print("received transcription")
    #    print(segments)

    # Log metrics and collect usage data
    def on_metrics_collected(agent_metrics: metrics.AgentMetrics):
        metrics.log_metrics(agent_metrics)
        usage_collector.collect(agent_metrics)

    #agent = Agent(
    #    instructions="You are a friendly voice assistant built by LiveKit."
    #)
    agent = Assistant()
    '''Agent(
        instructions="You are a friendly voice assistant built by LiveKit.",
        stt=stt,
        tts=tts
    )'''

    print("AWS-->", os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_REGION"))

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        # minimum delay for endpointing, used when turn detector believes the user is done with their turn
        min_endpointing_delay=0.5,
        # maximum delay for endpointing, used when turn detector does not believe the user is done with their turn
        max_endpointing_delay=5.0
    )

    # Trigger the on_metrics_collected function when metrics are collected
    session.on("metrics_collected", on_metrics_collected)

    try:
        logger.info("Waiting for participant to join (max 30s)...")
        participant = await asyncio.wait_for(ctx.wait_for_participant(), timeout=30)
        logger.info(f"Participant joined: {participant.identity}")
    except asyncio.TimeoutError:
        logger.warning("No participant joined within 30 seconds. Shutting down agent.")
        await ctx.close()
        return  # Exits the entrypoint, safely ends the subprocess

    await session.start(room=ctx.room, agent=agent)

    #ctx.room.on("transcription_received", on_transcription_received)

    #await session.say("How are you doing today?")
    #transcript = session.stt.

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )