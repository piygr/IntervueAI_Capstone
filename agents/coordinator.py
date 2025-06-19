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
from utils.memory import MemoryManager, ConversationItem, InterviewPlan

import os
import time
from agents.interviewer import Interviewer

from livekit.agents import UserInputTranscribedEvent
from livekit.agents import ConversationItemAddedEvent
    

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
    def __init__(self, *, participant_identity: str, interview_plan: dict):
        super().__init__(
            instructions="not-needed",
            stt=stt,
            tts=tts
        )
        self.participant_identity = participant_identity
        self.interview_plan = InterviewPlan(**interview_plan)
        self.memory = MemoryManager(session_id=participant_identity, 
                                    memory_dir="logs/memory",
                                    interview_plan=self.interview_plan)
        logger.info(self.memory)
        self.current_qn_index = -1
        self.agent_last_conversation = 0
        self.user_last_coversation = 0
        
    
    async def on_user_input_transcribed(self, user_transcript):
        print("on_user_input_transcribed:", user_transcript)
        if self.user_last_coversation == 0:
            self.memory.update_candidate_response(
                question_index=self.current_qn_index, 
                response=user_transcript,
                timestamp=time.time()
            )
            self.user_last_coversation = time.time()

            agent_response = await self.interviewer.decide_next_action(self.memory)
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
                
                print("Is session valid?", self.session)
                message = agent_response.get('agent_action_message')
                print("About to say:", message)
                await self.session.say(message)
                print("Finished saying:", message)

        #raise StopResponse()


    def handle_transcription_event(self, event: UserInputTranscribedEvent):
        print(f"User input transcribed: {event.transcript}, final: {event.is_final}")
        if event.is_final:
            self.session.say("Test message")
            #self.tts
            #print("Task created!!")
            user_transcript = event.transcript
            
            #async def safe_handle():
            #    await self.on_user_input_transcribed(user_transcript=user_transcript)

            #safe_handle()

        

    async def on_enter(self):
        #self.session.on("user_input_transcribed", self.handle_transcription_event)

    
        self.start_time = time.time()
        self.interviewer = Interviewer(prompt_file_path="prompts/interviewer.txt")
        initial_response = await self.interviewer.decide_next_action(self.memory, interview_phase="START")
        logger.info(initial_response)
        item = ConversationItem(
            type=initial_response.get('agent_action'),
            question_index=-1,
            time_spent_on_question_in_minutes= int( (time.time() - self.start_time) / 60 ),
            agent=initial_response.get('agent_action_message'),
            timestamp=time.time(),
            question_status="",
            question_remarks=""
        )
        self.agent_last_conversation = time.time()
        self.memory.add_conversation_item(item)
        self.session.say(initial_response.get('agent_action_message'), allow_interruptions=False)
        
    
    

    async def on_user_turn_completed(self, chat_ctx: llm.ChatContext, new_message: llm.ChatMessage):
        user_transcript = new_message.text_content
        logger.info(f"User {self.participant_identity} said -> {user_transcript}")
        self.memory.update_candidate_response(
                question_index=self.current_qn_index, 
                response=user_transcript,
                timestamp=time.time()
            )
        self.user_last_coversation = time.time()
        time.sleep(2)

        logger.info(f"user: {self.session.user_state}, agent: {self.session.agent_state}")
        if self.session.agent_state not in ['speaking'] and self.session.user_state not in ['speaking']:
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