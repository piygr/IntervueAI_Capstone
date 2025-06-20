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
    ModelSettings
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
        prompt_file_path = "prompts/coordinator.txt"
        
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
            tools=[]
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
            return full_output
        

    async def on_enter(self):
        self.session.generate_reply(instructions="Greet the candidate and wish luck for the interview")
    
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