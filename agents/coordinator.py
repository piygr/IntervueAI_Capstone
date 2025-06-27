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

from agents.feedback_agent import FeedbackAgent
from utils.session import fetch_session
from utils.memory import MemoryManager, ConversationItem, InterviewPlan

import os, json
import random
import time
from agents.interviewer import Interviewer

from livekit.agents import UserInputTranscribedEvent
from livekit.agents import ConversationItemAddedEvent
from utils.llm import extract_json_string
from dataclasses import dataclass, field
from typing import Optional

from livekit.api import LiveKitAPI, ListParticipantsRequest, RoomParticipantIdentity

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

@dataclass
class InterviewContext:
    interview_plan: InterviewPlan = None
    response_summary: list[dict] = field(default_factory=list)
    current_question_index: int = -1
    agent_last_conversation: int = 0
    user_last_conversation: int = 0
    user_last_typed: int = 0
    current_question_requires_coding: bool = False
    cancellation_warned: bool = False

silence_break_time_for_coding = 5*60 # 5 min

SILENCE_BREAKER = [
  "Are you still with me?",
  "Take your time, but let me know if you’re stuck.",
  "Just checking in — is everything okay on your side?",
  "Feel free to think out loud if that helps.",
  "No rush, but I’m here if you need a hint or clarification.",
  "Do you want me to repeat the question?",
  "Hey, let me know when you’re ready to continue.",
  "Just making sure we’re still connected.",
  "Would you like to hear the question again?",
  "Don’t worry, I’m here to support you — let me know if you need anything.",
  "It’s okay to pause, but I’m still listening if you’re ready.",
  "If you’re thinking, take your time — I just wanted to check in.",
  "Is there anything unclear that I can help with?",
  "Sometimes thinking aloud can help — feel free to do that.",
  "Would a small nudge help you move forward?",
  "No pressure — just let me know if you’re ready to proceed.",
  "Just a gentle ping — are you still working on the question?",
  "I’m happy to wait, but let me know if you need support.",
  "You’ve been quiet for a bit — should we move on or dig deeper?",
  "If you're lost in thought, that's perfectly fine. Just checking if you're still there."
]

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
        self.user_last_conversation = 0
        self.silence_watchdog_task = None
        self._cancel_interview_task = None
        self._lock = asyncio.Lock()
    
    '''async def llm_node(
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
            return ""'''
        

    async def on_enter(self):
        self.session.on("user_input_transcribed", self.user_input_transcribed)
        self.silence_watchdog_task = asyncio.create_task(self._monitor_silence())
        self.session.userdata.interview_plan = self.interview_plan
        self.session.userdata.user_last_conversation = time.time()
        self.session.generate_reply(allow_interruptions=False)
        
        
    def user_input_transcribed(self, event: UserInputTranscribedEvent):
        if event.is_final:
            logger.info(f"📝 User input transcribed: {event.transcript}, final: {event.is_final}")
            self.user_last_conversation = time.time()

            self.session.userdata.user_last_conversation = time.time()

            if self.session.userdata.current_question_index >= 0:
                self.memory.update_candidate_response(
                    question_index=self.session.userdata.current_question_index, 
                    response=event.transcript,
                    timestamp=time.time()
                )

    async def _monitor_silence(self):
        logger.info(f"👂 Started monitor silence")
        try:
            while True:
                await asyncio.sleep(10)

                # If agent or user is currently speaking, skip
                # If both have been silent for over 10 seconds
                #async with self._lock:
                now = time.time()
                logger.info(f"⌛️ agent_last_conversation: {self.session.userdata.agent_last_conversation}, user_last_conversation: {self.session.userdata.user_last_conversation}")
                _silence_break_time = silence_break_time_for_coding if self.session.userdata.current_question_requires_coding else random.randint(15, 20)

                logger.info(f'⏱️ silence break time: {_silence_break_time}')

                if self._cancel_interview_task is None and \
                now - max(self.session.userdata.agent_last_conversation, self.session.userdata.user_last_conversation) > _silence_break_time:
                        msg = random.choice(SILENCE_BREAKER)
                        await self.session.say(msg)
                        self.session.userdata.agent_last_conversation = now

                        warn_wait_time = 2 * silence_break_time_for_coding if self.session.userdata.current_question_requires_coding else 60
                        logger.info(f'⏱️ warn_wait_time: {warn_wait_time}')
                        if not self.session.userdata.cancellation_warned and now - self.session.userdata.user_last_conversation > warn_wait_time:
                            self.session.userdata.cancellation_warned = True
                            await self.session.say("You've been silent for a long time. Your interview will be cancelled soon. Please speak up to avoid cancellation.")

                        cancel_wait_time = 3 * silence_break_time_for_coding if self.session.userdata.current_question_requires_coding else 120
                        logger.info(f'⏱️ cancel_wait_time: {cancel_wait_time}')
                        if now - self.session.userdata.user_last_conversation > cancel_wait_time:
                            cancel_message = "Looks like you are away, I am cancelling the interview for now. Please connect again later."
                            self._cancel_interview_task = await self._cancel_interview(cancel_message)
                            break
                        
        except Exception as e:
            logger.error(f"Error in _monitor_silence {e}")

    async def _cancel_interview(self, message: str):
        #async with self._lock:
        async with LiveKitAPI() as lkapi:
            if message:
                await self.session.say(message, allow_interruptions=False)

            await FeedbackAgent.generate_feedback(self.memory)
            interview_room = get_job_context().room
            await interview_room.local_participant.publish_data(
                b"interview-ended",
                reliable=True,
                topic="interview-status"
            )

            res = await lkapi.room.list_participants(ListParticipantsRequest(
                room=interview_room.name
            ))
            for p in res.participants:
                await lkapi.room.remove_participant(RoomParticipantIdentity(
                    room=interview_room.name,
                    identity=p.identity,
                ))
            await self.silence_watchdog_task.cancel()
            await self.session.drain()
            await self.session.aclose()

    @function_tool
    async def greet(self,
        context: RunContext[InterviewContext],
        greeting_message: str
    ):
        """
        Called to greet the candidate and wish him/her luck before the interview starts
        
        Args:
            greeting_message: This is the greetig message which will be conveyed by the voice agent. 
        """
        logger.info(f"💬 {greeting_message}")
        await self.session.say(greeting_message)
        context.userdata.agent_last_conversation = time.time()


    @function_tool
    async def stay_silent(self,
        context: RunContext[InterviewContext],
        current_question_index: int
    ):
        """
        Called when the candidate may still be speaking or thinking, as an interviewer you don't have any message for the candidate so better to stay silent.
        
        Args:
            current_question_index: Index of the current active question from the interview plan
        """

        logger.info(f"💬 {current_question_index}: Stay Silent")
        context.userdata.current_question_index = current_question_index

    @function_tool
    async def probe(
        self,
        context: RunContext[InterviewContext],
        current_question_index: int,
        probing_message: str
    ):
        """
        Called to ask a follow-up question to explore the answer more deeply, depending on the interview plan's evaluation depth and the previous user response.

        Args:
            current_question_index: Index of the current active question from the interview plan
            probing_question: Message that needs to be asked to the candidate to probe further. Make sure the message blends in flow of the conversation. 
        """

        logger.info(f"💬 {current_question_index}: Probe Further: {probing_message}")
        context.userdata.current_question_index = current_question_index
        await self.session.say(probing_message)
        context.userdata.agent_last_conversation = time.time()

        self.memory.add_followup_item(question_index=current_question_index,
                                            followup_type="probe_further",
                                            followup=probing_message,
                                            timestamp=time.time())


    @function_tool
    async def hint(
        self,
        context: RunContext[InterviewContext],
        current_question_index: int,
        hint_message: str
    ):
        """
        Called to offer a helpful clue or nudge if the candidate seems stuck or confused.
        
        Args:
            current_question_index: Index of the current active question from the interview plan
            hint_message: Hint message that needs to be told to the candidate. Make sure the message blends in flow of the conversation.
        """

        logger.info(f"💬 {current_question_index}: Provide hint: {hint_message}")
        context.userdata.current_question_index = current_question_index
        await self.session.say(hint_message)
        context.userdata.agent_last_conversation = time.time()

        self.memory.add_followup_item(question_index=current_question_index,
                                            followup_type="provide_hint",
                                            followup=hint_message,
                                            timestamp=time.time())


    @function_tool
    async def clarify(
        self,
        context: RunContext[InterviewContext],
        current_question_index: int,
        clarification_message: str
    ):
        """
        Called to rephrase or repeat the question if the candidate misunderstood or asked for clarification

        Args:
            current_question_index: Index of the current active question from the interview plan
            clarification_message: Clarification message that needs to be told to the candidate. Make sure the message blends in flow of the conversation.
        """
        logger.info(f"💬 {current_question_index}: Clarification message: {clarification_message}")
        context.userdata.current_question_index = current_question_index
        await self.session.say(clarification_message)
        context.userdata.agent_last_conversation = time.time()

        self.memory.add_followup_item(question_index=current_question_index,
                                            followup_type="clarify",
                                            followup=clarification_message,
                                            timestamp=time.time())
        

    @function_tool
    async def next_question(self,
        context: RunContext[InterviewContext],
        question_index: int,
        pre_message: str,
        question_message: str,
        previous_question_user_response_summary: str,
        use_code_editor: bool
    ):
        """
        Called to move to the next question once the current one has been sufficiently answered

        Args:
            question_index: Index of the question from the interview plan being asked in the question_message. index starts from 0 for first question and so on.
            pre_message: Message that is conveyed to the candidate before asking the next question. Examples: 
                        - Let's move on to the next question.
                        - Moving on to the next question.
                        - Alright. Let's discuss something else. 

            question_message: Next question message that needs to be asked to the candidate. Make sure the message blends in flow of the conversation.
            previous_question_user_response_summary: Detailed summary of the user response for the previous question
            use_code_editor: If the question is a coding related question or DSA or algo related question which requires code editor use, make it true otherwise false.
        """

        logger.info(f"💬 {question_index}: Next Question message: {question_message}")
        context.userdata.current_question_index = question_index
        if question_index > 0:
            context.userdata.response_summary.append(dict(question_index=question_index, 
                                                          user_response_summary=previous_question_user_response_summary))
            
        context.userdata.current_question_requires_coding = use_code_editor
        if not use_code_editor:
            await self.session.say(pre_message + "\n" + question_message)
        else:
            question = self.interview_plan.questions[question_index].question
            logger.info(f"💬 coding question: {question}")
            room = get_job_context().room
            await room.local_participant.publish_data(
                question.encode("utf-8"),
                reliable=True,
                topic="code-editor"
            )
            await self.session.say(pre_message + "\n" + question_message + "\n Please open the code editor to answer.")

        context.userdata.agent_last_conversation = time.time()

        self.memory.add_agent_question(question_index=question_index,
                                            agent=question_message,
                                            timestamp=time.time())

        #logger.info(self.memory.get_conversation())

    @function_tool
    async def end_interview(self,
        context: RunContext[InterviewContext],
        last_question_index: int,
        end_interview_message: str,
        previous_question_user_response_summary: str,
    ):
        """
        Called to end the interview due to time constraint or all questions have been asked.

        Args:
            last_question_index: Index of the last question from the interview plan after which the end of interview is reached
            
            end_interview_message: Message to convey to the candidate at the end of interview. Make sure the message blends in flow of the conversation.
            previous_question_user_response_summary: Detailed summary of the user response for the previous question
        """
        logger.info(f"💬 {last_question_index}: End of interview, message: {end_interview_message}")
        
        context.userdata.response_summary.append(dict(question_index=last_question_index, 
                                                          user_response_summary=previous_question_user_response_summary))
        await self.session.say(end_interview_message)
        context.userdata.agent_last_conversation = time.time()
        await self._cancel_interview(message=None)

    async def handle_data_received(self, packet):
        # packet is a livekit.rtc.DataPacket
        data = packet.data
        topic = packet.topic
        participant = packet.participant.identity  # if you need it
        # logger.debug(f"📦 Received {len(data)} bytes on topic {topic}")
        if topic == "code-editor":
            code = data.decode("utf-8")

            marker = "// Type your code below this:"
            parts = code.split(marker, 1)
            if len(parts) > 1:
                self.session.userdata.user_last_typed = time.time()
                code = parts[1].strip()
                if code:
                    final_marker = "<--final code-->"
                    if code.strip().endswith(final_marker):
                        code = code.rstrip().removesuffix(final_marker).rstrip()
                        if code:
                            self.session.userdata.user_last_conversation = time.time()
                            logger.debug(f"📝 Code: {code}")
                            self.memory.update_candidate_response(
                                question_index=self.session.userdata.current_question_index, 
                                response=f"[Candidate submitted the following code]\n '''\n{code}'''\n]",
                                timestamp=time.time()
                            )
                            # await self.session.say("Thank you for the submission.")
                            code_submission_prompt = "Candidate has submitted the code. Evaluate the code correctness, check if the code handles the edge cases correctly and ask further probing question on it including time and space complexity if it has not been answered already in the code text."
                            handle = self.session.generate_reply(user_input=code, instructions=code_submission_prompt)
                            await handle

                        else:
                            await self.session.say("Your submission is empty. Please submit full code")
                    else:
                        logger.debug(f"📝 streaming code: {code}")
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
        self.user_last_conversation = time.time()
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
