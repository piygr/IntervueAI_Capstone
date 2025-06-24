import logging

from dotenv import load_dotenv
import asyncio
from collections.abc import AsyncGenerator, AsyncIterable, Coroutine, Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

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
    utils,
    ModelSettings,
    RunContext,
    function_tool,
    get_job_context
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
from utils.memory import MemoryManager, ConversationItem, InterviewPlan

import os, json
import random
import time
from agents.interviewer import Interviewer

from livekit.agents import UserInputTranscribedEvent
from livekit.agents import ConversationItemAddedEvent
from utils.llm import extract_json_string

logger = logging.getLogger(__name__)
load_dotenv(dotenv_path=".env.local")

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

class Coordinator(Agent):
    def __init__(self, *, participant_identity: str, interview_plan: dict, interview_context: dict):
        
        self.system_prompt = ""
        prompt_file_path = "prompts/coordinator.3.txt"
        
        if os.path.exists(prompt_file_path):
            with open(prompt_file_path, 'r') as f:
                self.system_prompt = f.read()

        self.prompt = self.system_prompt.format(interview_plan=interview_plan,
                                                **interview_context)

        super().__init__(
            instructions=self.prompt,
            stt=stt,
            tts=tts,
            llm=llm_model
        )

        self.participant_identity = participant_identity
        self.interview_plan = InterviewPlan(**interview_plan)
        self.interview_context = interview_context
        self.memory = MemoryManager(session_id=participant_identity, 
                                    memory_dir="logs/memory",
                                    interview_plan=self.interview_plan)
        
        self.current_qn_index = -1
        self.agent_last_conversation = 0
        self.user_last_coversation = 0
        self.silence_watchdog_task = None
        self._cancel_interview_task = None
        self._lock = asyncio.Lock()
    
    async def llm_node(
        self,
        chat_ctx: ChatContext,
        tools: list = [],
        model_settings: ModelSettings = None
    ):# -> Coroutine[Any, Any, llm.ChatChunk]:
        # Compose structured system prompt
        
       #chat_ctx.add_message(
        #    content=self.prompt,
        #    role="system"
        #)

        #chat_ctx.

        # Add LLM-callable tools if needed
        #all_tools = tools + [summarize_tool, end_interview]

        full_output = ""
        # Call the LLM and yield result
        stream = self.llm.chat(
            chat_ctx=chat_ctx,
            tools=tools
        )
        
        async for chunk in stream:
            if isinstance(chunk, llm.ChatChunk):
                if chunk.delta and chunk.delta.content:
                    full_output += chunk.delta.content
            else:
                full_output += str(chunk)

        try:
            logger.info(full_output)
            if full_output:
                json_object = json.loads(extract_json_string(full_output))
                if json_object.get('use_code_editor_tool'):
                    logger.info(f'🛠️ need to call tool here...')
                    ques_index = json_object.get('current_question_index')
                    ques = self.interview_plan.questions[ques_index].question
                    await self.send_question_to_code_editor(ques)
                    # await self.session.say("please open the code editor now.")
                    return json_object.get('agent_action_message') + "please open the code editor now."
                if json_object.get('agent_action') not in ['stay_silent'] \
                    or (time.time() - self.agent_last_conversation) > 10:
                    self.agent_last_conversation = time.time()
                    return json_object.get('agent_action_message')
                else:
                    return ""
            else:
                return ""
        except Exception as e:
            logger.error(f"Error in parsing json:{e}")
            return ""
        

    async def on_enter(self):
        self.session.on("user_input_transcribed", self.user_input_transcribed)
        self.session.generate_reply(instructions="Greet the candidate and wish him/her luck for the interview")
        self.silence_watchdog_task = asyncio.create_task(self._monitor_silence())
        
    #async def on_user_turn_completed(self, chat_ctx: llm.ChatContext, new_message: llm.ChatMessage):
        #self.user_last_coversation = time.time()

    def user_input_transcribed(self, event: UserInputTranscribedEvent):
        if event.is_final:
            print(f"User input transcribed: {event.transcript}, final: {event.is_final}")
            self.user_last_coversation = time.time()

    async def _monitor_silence(self):
        #while True:
        await asyncio.sleep(10)

        # If agent or user is currently speaking, skip
        # If both have been silent for over 10 seconds
        #async with self._lock:
        now = time.time()
        if self._cancel_interview_task is None and \
        now - max(self.agent_last_conversation, self.user_last_coversation) > random.randint(15, 20):
            #l = ["Are you there?", "Could you please tell me what you are thinking?", "Let me know if you need clarification."]
            #msg = l[random.randint(0, len(l)-1)]
            silence_break_prompt = f"Candidate has not spoken for a while, ask him if he/she is thiking or he has lost the connection or needs clarification."
            await self.session.generate_reply(instructions=silence_break_prompt)
            self.agent_last_conversation = now
            self.silence_watchdog_task = asyncio.create_task(self._monitor_silence())
        
                #if now - self.user_last_coversation > 60:
                #    self._cancel_interview_task = asyncio.create_task(self._cancel_interview)

    async def _cancel_interview(self):
        #async with self._lock:
        await self.session.say("Looks like you are away, I am cancelling the interview for now. Please connect again later.", allow_interruptions=False)
        self.silence_watchdog_task.cancel()
        self.session.aclose()

    # @function_tool()
    async def send_question_to_code_editor(self, question: str):
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
                            code_submission_prompt = "User has submitted the code. Evaluate the code correctness and ask further question on it."
                            handle = self.session.generate_reply(user_input=code, instructions=code_submission_prompt)
                            await handle
                        else:
                            await self.session.say("Your submission is empty. Please submit full code")
                    else:
                        logger.info(f"💬 streaming code: {code}")
            else:
                await self.session.say("Please do not delete any existing code or text from the editor.")

    '''async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> (
        AsyncGenerator[rtc.AudioFrame, None]
        | Coroutine[Any, Any, AsyncIterable[rtc.AudioFrame]]
        | Coroutine[Any, Any, None]
    ):
        llm_resp = ""
        async for chunk in text:
            llm_resp += chunk
        
        message = ""
        try:
            if llm_resp:
                llm_resp_json = json.loads(extract_json_string(llm_resp))
                message = llm_resp_json.get("agent_action_message")
        except Exception as e:
            logger.error(f"Error in json parsing in tts_node: {e}")

        logger.info(f"tts_node: {message}")
        yield message'''

"""
    async def on_user_turn_completed(self, chat_ctx: llm.ChatContext, new_message: llm.ChatMessage):
        #async with self._lock:
        user_transcript = new_message.text_content
        logger.info(f"User {self.participant_identity} said -> {user_transcript}")
        '''self.memory.update_candidate_response(
                question_index=self.current_qn_index, 
                response=user_transcript,
                timestamp=time.time()
            )'''
        self.user_last_coversation = time.time()
        #time.sleep(2)
        '''self.update_instructions(instructions=self.system_prompt.format(interview_plan=self.interview_plan,
                                                                        conversation=self.memory.to_json().get("conversation"),
                                                                        current_question=self.current_qn_index))
        
        

        logger.info(f"user: {self.session.user_state}, agent: {self.session.agent_state}")
        #if self.session.agent_state not in ['speaking'] and self.session.user_state not in ['speaking']:
        agent_response = await self.interviewer.decide_next_action(self.memory)
        #probe_further, provide_hint, clarify, move_to_next_question, end_interview
        if agent_response.get('agent_action') == 'stay_silent':
            pass
        else:
            if agent_response.get('agent_action') == "move_to_next_question":
                self.current_qn_index += 1
                self.memory.add_agent_question(question_index=self.current_qn_index,
                                            agent=agent_response.get('agent_action_message'),
                                            timestamp=time.time())
            else:
                self.memory.add_followup_item(question_index=self.current_qn_index,
                                            followup_type=agent_response.get('agent_action'),
                                            followup=agent_response.get('agent_action_message'),
                                            timestamp=time.time())
            
            self.session.say(agent_response.get('agent_action_message'))
        raise StopResponse()
        '''
        """
