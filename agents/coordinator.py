import logging

from dotenv import load_dotenv
import asyncio
from dataclasses import dataclass

from livekit import rtc
from livekit.agents import (
    Agent,
    RunContext,
    function_tool,
    get_job_context
)

from livekit.plugins import (
    openai,
    silero,
    aws,
    google
)
#from livekit.plugins.turn_detector.english import EnglishModel

from agents.feedback_agent import FeedbackAgent
from utils.memory import MemoryManager, ConversationItem, InterviewPlan

import os
import random
import time
import json

from livekit.agents import UserInputTranscribedEvent
from utils.session import load_config, get_google_api_key

from dataclasses import dataclass, field

from livekit.api import LiveKitAPI, ListParticipantsRequest, RoomParticipantIdentity

logger = logging.getLogger(__name__)
load_dotenv(dotenv_path=".env.local")
config = load_config()
    
stt = aws.STT()
tts = aws.TTS(
    voice=config.get('tts', {}).get('voice', ''),
    language=config.get('tts', {}).get('language', ''),
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
    last_question_index: int = -1
    timetracker: dict = field(default_factory=dict)
    agent_last_conversation: int = 0
    user_last_conversation: int = 0
    user_last_typed: int = 0
    current_question_requires_coding: bool = False
    cancellation_warned: bool = False
    
silence_break_time_for_coding = 5*60 # 5 min


SILENCE_BREAKER = {
    "question": [
        "Are you still with me?",
        "Take your time, but let me know if you’re stuck.",
        "Just checking in — is everything okay on your side?",
        "Feel free to think out loud if that helps.",
        "No rush, but I’m here if you need a hint or clarification.",
        "Do you want me to repeat the question?",
        "Hey, let me know when you’re ready to continue.",
        "Just making sure we’re still connected.",
        "Would you like to hear the question again?",
        "Do you need help — let me know if you need anything.",
        "It’s okay to pause, but I’m still listening if you’re ready.",
        "If you’re thinking, take your time — I just wanted to check in.",
        "Is there anything unclear that I can help with?",
        "Sometimes thinking aloud can help — feel free to do that.",
        "Would a small nudge help you move forward?",
        "No pressure — just let me know if you’re ready to proceed.",
        "Just a gentle ping — are you still working on the question?",
        "I’m happy to wait, but let me know if you need help.",
        "You’ve been quiet for a bit — should we move on or dig deeper?",
        "If you're lost in thought, that's perfectly fine. Just checking if you're still there."
    ],
    "greet": [
        "Just checking — can you hear me clearly?",
        "Whenever you’re ready, feel free to say hello so we can begin.",
        "No rush — take your time and let me know when you’re ready to start.",
        "I'm here and ready when you are!",
        "Shall we get started with the interview?",
        "Hi again! Just waiting for you to say something so we can begin.",
        "If you're facing any tech issues, feel free to reload or unmute.",
        "Is everything working okay on your end?",
        "Feel free to say 'ready' when you're good to go!",
        "Just a friendly nudge — we can begin whenever you're set.",
        "Let me know if you need a moment or if you'd like to start now.",
        "I haven’t heard from you yet — is your mic working?",
        "Ready to dive in? You can speak when you're set.",
        "I'm all ears — shall we begin?",
        "Still here with you. Say something when you're ready!",
        "Looks like we haven’t started yet — everything alright?",
        "Give me a quick hello so I know we can proceed.",
        "No pressure — just let me know when you're ready to kick things off.",
        "We haven’t started yet — is there anything you need help with before we begin?",
        "Say something when you're ready, and we’ll get started right away!"
    ]

}

class Coordinator(Agent):
    def __init__(self, *, participant_identity: str, interview_plan: dict, interview_context: dict):
        
        self.system_prompt = ""
        prompt_file_path = "prompts/coordinator.txt"
        
        if os.path.exists(prompt_file_path):
            with open(prompt_file_path, 'r') as f:
                self.system_prompt = f.read()

        self.prompt = self.system_prompt.format(interview_plan=interview_plan,
                                                **interview_context)

        llm_model = self.load_llm_model()

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
        
        self.silence_watchdog_task = None
        self._cancel_interview_task = None
        self._lock = asyncio.Lock()
        self._user_disconnected = False
        

    def load_llm_model(self):
        
        llm_model = None
        if config.get('model', {}).get('type', '') == 'gemini':
            
            google_api_key, _ = get_google_api_key()
            llm_model = google.LLM(
                    model=config.get('model', {}).get('name', ''),
                    api_key=google_api_key
                )
            
        elif config.get('model', {}).get('type', '') == 'ollama':
            llm_model = openai.LLM.with_ollama(
                model=config.get('model', {}).get('name', ''),
                base_url=config.get('model', {}).get('url', ''),
            )

        return llm_model
    

    async def on_enter(self):
        self.session.userdata.user_last_conversation = time.time()
        self.session.on("user_input_transcribed", self.user_input_transcribed)
        self.session._room_io._room.on("participant_disconnected", self.user_disconnected)
        self.silence_watchdog_task = asyncio.create_task(self._monitor_silence())
        self.session.userdata.interview_plan = self.interview_plan
        self.session.userdata.user_last_conversation = time.time()
        self.session.generate_reply(allow_interruptions=False)
        
        
    def user_input_transcribed(self, event: UserInputTranscribedEvent):
        if event.is_final:
            logger.info(f"📝 User input transcribed: {event.transcript}, final: {event.is_final}")

            self.session.userdata.user_last_conversation = time.time()

            if self.session.userdata.current_question_index >= 0:
                self.memory.update_candidate_response(
                    question_index=self.session.userdata.current_question_index, 
                    response=event.transcript,
                    timestamp=time.time()
                )

    def user_disconnected(self):
        self._user_disconnected = True


    async def _monitor_silence(self):
        logger.info(f"👂 Started monitor silence")
        try:
            while True:
                await asyncio.sleep(20)

                now = time.time()
                
                async with self._lock:
                    question_index = self.session.userdata.current_question_index
                    current_question_start_time = self.session.userdata.timetracker.get(question_index, 0)
                    if question_index >= 0 and current_question_start_time > 0:
                        time_spent_so_far = "{:.2f}".format( (now - current_question_start_time) / 60.0)
                        logger.info(f"moitor_silence: time_spent_so_far - {time_spent_so_far}")
                        self.session._chat_ctx.add_message(role="system", content=f"⏱Total Time spent on Ongoing Question (Question Index: {question_index} ) so far = ** {time_spent_so_far} minutes **")
                   
                logger.info(f"⌛️ agent_last_conversation: {self.session.userdata.agent_last_conversation}, user_last_conversation: {self.session.userdata.user_last_conversation}")
                _silence_break_time = random.randint(200, 300) if self.session.userdata.current_question_requires_coding else random.randint(15, 20)

                logger.info(f'⏱️ silence break time: {_silence_break_time}')

                if self._cancel_interview_task is None and \
                now - max(self.session.userdata.agent_last_conversation, self.session.userdata.user_last_conversation) > _silence_break_time:
                        if self.session.userdata.current_question_index >= 0:
                            msg = random.choice(SILENCE_BREAKER['question'])
                        else:
                            msg = random.choice(SILENCE_BREAKER['greet'])
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

                if self._user_disconnected:
                    cancel_message = "Looks like you are away, I am cancelling the interview for now. Please connect again later."
                    self._cancel_interview_task = await self._cancel_interview(cancel_message)
                    break
                        
        except Exception as e:
            logger.error(f"Error in _monitor_silence {e}")

    async def _cancel_interview(self, message: str):
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
        But make sure staying silent is not creating a deadlock.
        
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
        

    # @function_tool
    # async def confirm_if_completed(
    #     self,
    #     context: RunContext[InterviewContext],
    #     current_question_index: int,
    #     confirm_message: str
    # ):
    #     """
    #     Called to confirm with the candidate if he/she has completed the response, and you can move on to ask the next question.
    #     Make sure this is not called every time before moving to the next question. It should be called in case candidate is still speaking
    #     and his / her response is not yet completed and enough probing is done with respect to the question and enough time has been spent on the question.

    #     Args:
    #         current_question_index: Index of the current active question from the interview plan.
    #         confirm_message: Confirmation message that needs to be asked to the candidate. Make sure the message blends in flow of the conversation.
    #     """

    #     logger.info(f"💬 {current_question_index}: Confirmation message: {confirm_message}")
    #     context.userdata.current_question_index = current_question_index
    #     await self.session.say(confirm_message)
    #     context.userdata.agent_last_conversation = time.time()

    #     self.memory.add_followup_item(question_index=current_question_index,
    #                                         followup_type="clarify",
    #                                         followup=confirm_message,
    #                                         timestamp=time.time())

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

        now = time.time()
        logger.info(f"💬 {question_index}: Next Question message: {question_message}")
        context.userdata.last_question_index = context.userdata.current_question_index
        context.userdata.current_question_index = question_index
        if question_index > 0:
            context.userdata.response_summary.append(dict(question_index=context.userdata.last_question_index, 
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

        context.userdata.agent_last_conversation = now

        self.memory.add_agent_question(question_index=question_index,
                                            agent=question_message,
                                            timestamp=now)

        async with self._lock:
            self.session.userdata.timetracker[question_index] = now
            if context.userdata.last_question_index >= 0:
                last_question_start_time = self.session.userdata.timetracker[context.userdata.last_question_index]
                time_spent_last_question =  "{:.2f}".format( (now - last_question_start_time) / 60.0)
                self.session._chat_ctx.add_message(role="system", content=f"⏱Total Time spent on Last Question (Question Index: {context.userdata.last_question_index} ) = ** {time_spent_last_question} minutes **")


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
        #self.session.say(end_interview_message)
        context.userdata.agent_last_conversation = time.time()
        await self._cancel_interview(message=end_interview_message)

  
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
                            # await self.session.say("Thank you for the submission.")
                            code_submission_prompt = "Candidate has submitted the code. Evaluate the code correctness, check if the code handles the edge cases correctly and ask further probing question on it including time and space complexity if it has not been answered already in the code text."
                            handle = self.session.generate_reply(user_input=code, instructions=code_submission_prompt)
                            await handle

                            self.memory.update_candidate_response(
                                question_index=self.session.userdata.current_question_index, 
                                response="[Candidate submitted the following code]\n '''\n{code}'''\n]",
                                timestamp=time.time()
                            )   

                        else:
                            await self.session.say("Your submission is empty. Please submit full code")
                    else:
                        logger.debug(f"📝 streaming code: {code}")
            else:
                await self.session.say("Please do not delete any existing code or text from the editor.")
