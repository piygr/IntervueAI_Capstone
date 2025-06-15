import logging
from typing import AsyncIterable, Optional

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
    llm,
    stt,
    FunctionTool
)
from livekit.agents.llm.chat_context import ChatContext, ChatMessage
from livekit.agents.llm import StopResponse
from livekit.agents.stt.stt import SpeechEvent
from livekit.agents.voice.agent import ModelSettings
from livekit.plugins import (
    openai,
    silero,
    aws,
    google,
    noise_cancellation
)
from livekit import rtc
from livekit.plugins.turn_detector.english import EnglishModel
from livekit.agents import UserStateChangedEvent, AgentStateChangedEvent

from utils.utils import fetch_session
from utils import llm as local_llm
from datetime import datetime
from google import genai

import os

load_dotenv(dotenv_path=".env.local")
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

logger = logging.getLogger("voice-agent")

my_llm = google.LLM(
        model="gemini-2.0-flash",
    )
# my_llm = google.beta.realtime.RealtimeModel(
#         model="gemini-2.0-flash-exp",
#         voice="Puck",
#         temperature=0.8,
#         instructions="You are a helpful voice AI assistant",
#     )

my_stt = aws.STT()
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
        instructions1 = "You are a helpful voice AI assistant."

        super().__init__(
            instructions=instructions,
            # stt=stt,
            # tts=tts,
            #turn_detection=EnglishModel,
            # llm=my_llm
        )

    async def on_enter(self):
        # The agent should be polite and greet the user when it joins :)
        # self.session.say("Hey, how are you doing today?")
        handle = self.session.say("Hey, how are you doing today?")
        handle.add_done_callback(lambda _: print("speech done"))

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        logger.info("user turn completed...")
        cur_time = datetime.now()
        logger.info(f'chat ctx:...{self.session._chat_ctx.items}...')
        logger.info(f'turn_ctx:----{turn_ctx.items}----')
        print(f'time elapsed: {cur_time - time}')

        user_transcript = new_message.text_content
        logger.info(f"user text: {user_transcript}")

        # prompt = f"Give me only a one sentence reply to these text: {user_transcript}"
        # reply = await local_llm.call_llm_with_timeout(client=client, prompt=prompt)
        # self.session.say(reply.strip())

        # self.session.say("I am listening...")
        # raise StopResponse()

        if ("wait for a minute" in new_message.text_content):
            print(f'ok waiting....')
            # await self.session.interrupt()
            turn_ctx.add_message(role="user", content=new_message.text_content)
            await self.update_chat_ctx(turn_ctx)
            await self.session.say("Ok I'll wait")
            raise StopResponse()
        # # callback before generating a reply after user turn committed
        # if not new_message.text_content:
        #     # for example, raise StopResponse to stop the agent from generating a reply
        #     logger.info("ignore empty user turn")

    async def stt_node(self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings):
        logger.info("stt_node...")
        return Agent.default.stt_node(self, audio, model_settings)
    
    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[FunctionTool],
        model_settings: ModelSettings
    ) -> AsyncIterable[llm.ChatChunk]:
        logger.info("llm_node...")
        return Agent.default.llm_node(self, chat_ctx, tools, model_settings)

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        logger.info("tts_node...")
        return Agent.default.tts_node(self, text, model_settings)
        

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):

    await ctx.connect(
        auto_subscribe=AutoSubscribe.AUDIO_ONLY,
    )
    logger.info(f"connecting to room {ctx.room.name}")
    logger.info(f"Interview ID: {os.getenv('INTERVIEW_ID')}")

    session_id = ctx.room.name.replace('interview-', '')

    ### Fetching session details eg. JD, Resume ####
    session_details = fetch_session(session_id)
    # logger.info(fetch_session(session_id))
    ### 

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

    jd_id = session_details.get("JD")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    jd_path = os.path.join(script_dir, "..", "job_description", f"{jd_id}.json")

    with open(jd_path, "r", encoding="utf-8") as file:
        jd_text = file.read()

    agent = Assistant(job_description=jd_text, resume=session_details.get("resume"))
    # agent = Assistant(job_description="", resume="")
    
    #print("AWS-->", os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_REGION"))

    session = AgentSession(
        # turn_detection=EnglishModel(),
        vad=ctx.proc.userdata["vad"],
        # minimum delay for endpointing, used when turn detector believes the user is done with their turn
        min_endpointing_delay=0.5,
        # maximum delay for endpointing, used when turn detector does not believe the user is done with their turn
        max_endpointing_delay=5.0,
        allow_interruptions=True,
        stt=my_stt,
        tts=tts,
        llm=my_llm
    )

    # Trigger the on_metrics_collected function when metrics are collected
    session.on("metrics_collected", on_metrics_collected)


    @session.on("user_state_changed")
    def on_user_state_changed(ev: UserStateChangedEvent):
        if ev.new_state == "speaking":
            logger.info("User started speaking")
        elif ev.new_state == "listening":
            logger.info("User stopped speaking")
            global time
            time = datetime.now()
        elif ev.new_state == "away":
            logger.info("User is not present (e.g. disconnected)")

    @session.on("agent_state_changed")
    def on_agent_state_changed(ev: AgentStateChangedEvent):
        if ev.new_state == "initializing":
            logger.info("Agent is starting up")
        elif ev.new_state == "idle":
            logger.info("Agent is ready but not processing")
        elif ev.new_state == "listening":
            logger.info("Agent is listening for user input")
        elif ev.new_state == "thinking":
            logger.info("Agent is processing user input and generating a response")
            # session.interrupt()
            # logger.info("Will take time to respond")

        elif ev.new_state == "speaking":
            logger.info("Agent started speaking")

    @session.on("user_speech_committed")
    def on_user_speech_committed(msg: llm.ChatMessage):
        logger.debug(f'user speech commited... {msg}')

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
            # enable background voice & noise cancellation, powered by Krisp
            # included at no additional cost with LiveKit Cloud
            noise_cancellation=noise_cancellation.BVC(),
            )
        )

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )