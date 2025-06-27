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

from utils.session import fetch_session
from utils.memory import MemoryManager, ConversationItem, InterviewPlan

import os
import random
import time

from livekit.agents import UserInputTranscribedEvent
from utils.session import load_config

from dataclasses import dataclass, field

from livekit.api import LiveKitAPI, ListParticipantsRequest, RoomParticipantIdentity

logger = logging.getLogger(__name__)
load_dotenv(dotenv_path=".env.local")
config = load_config()

llm_model = None
if config.get('model', {}).get('type', '') == 'gemini':
    llm_model = google.LLM(
            model="gemini-2.0-flash",
        )
    
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
    agent_last_conversation: int = 0
    user_last_conversation: int = 0
    user_last_typed: int = 0
    current_question_requires_coding: bool = False


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
        

    async def on_enter(self):
        self.session.userdata.user_last_conversation = time.time()
        self.session.on("user_input_transcribed", self.user_input_transcribed)
        self.silence_watchdog_task = asyncio.create_task(self._monitor_silence())
        self.session.userdata.interview_plan = self.interview_plan
        self.session.generate_reply(allow_interruptions=False)
        
        
    def user_input_transcribed(self, event: UserInputTranscribedEvent):
        if event.is_final:
            print(f"User input transcribed: {event.transcript}, final: {event.is_final}")
            self.session.userdata.user_last_conversation = time.time()

            if self.session.userdata.current_question_index >= 0:
                self.memory.update_candidate_response(
                    question_index=self.session.userdata.current_question_index, 
                    response=event.transcript,
                    timestamp=time.time()
                )

    async def _monitor_silence(self):
        logger.info(f"Started monitor silence")
        try:
            while True:
                await asyncio.sleep(10)

                now = time.time()
                logger.info(f"agent_last_conversation: {self.session.userdata.agent_last_conversation}, user_last_conversation: {self.session.userdata.user_last_conversation}")
                if self._cancel_interview_task is None and \
                ((now - max(self.session.userdata.agent_last_conversation, self.session.userdata.user_last_conversation) > random.randint(40, 60) ) \
                    or (self.session.userdata.current_question_requires_coding and now - self.session.userdata.user_last_typed > random.randint(200, 300) ) ):
                        if self.session.userdata.current_question_index >= 0:
                            msg = random.choice(SILENCE_BREAKER['question'])
                        else:
                            msg = random.choice(SILENCE_BREAKER['greet'])

                        await self.session.say(msg)
                        self.session.userdata.agent_last_conversation = now
                        if now - self.session.userdata.user_last_conversation > 300:
                            msg = "Looks like you are away, I am cancelling the interview for now. Please connect again later."
                            self._cancel_interview_task = await self._cancel_interview(msg)
                            break
                        
        except Exception as e:
            logger.error(f"Error in _monitor_silence {e}")


    async def _cancel_interview(self, msg=""):
        async with LiveKitAPI() as lkapi:
            if msg:
                await self.session.say(msg, allow_interruptions=False)
            
            interview_room = get_job_context().room
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
        logger.info(f"{greeting_message}")
        self.session.say(greeting_message)
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

        logger.info(f"{current_question_index}: Stay Silent")
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

        logger.info(f"{current_question_index}: Probe Further: {probing_message}")
        context.userdata.current_question_index = current_question_index
        self.session.say(probing_message)
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

        logger.info(f"{current_question_index}: Provide hint: {hint_message}")
        context.userdata.current_question_index = current_question_index
        self.session.say(hint_message)
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
        logger.info(f"{current_question_index}: Clarification message: {clarification_message}")
        context.userdata.current_question_index = current_question_index
        self.session.say(clarification_message)
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

        logger.info(f"{question_index}: Next Question message: {question_message}")
        context.userdata.current_question_index = question_index
        if question_index > 0:
            context.userdata.response_summary.append(dict(question_index=question_index, 
                                                          user_response_summary=previous_question_user_response_summary))
            
        context.userdata.current_question_requires_coding = use_code_editor
        if not use_code_editor:
            self.session.say(pre_message + "\n" + question_message)
        else:
            question = self.interview_plan.questions[question_index].question
            logger.info(f"💬 coding question: {question}")
            room = get_job_context().room
            await room.local_participant.publish_data(
                question.encode("utf-8"),
                reliable=True,
                topic="code-editor"
            )
            self.session.say(pre_message + "\n" + question_message + "\n Please open code editor to answer.")

        context.userdata.agent_last_conversation = time.time()

        self.memory.add_agent_question(question_index=question_index,
                                            agent=question_message,
                                            timestamp=time.time())


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
        logger.info(f"{last_question_index}: End of interview")
        
        context.userdata.response_summary.append(dict(question_index=last_question_index, 
                                                          user_response_summary=previous_question_user_response_summary))
        #self.session.say(end_interview_message)
        context.userdata.agent_last_conversation = time.time()
        await self._cancel_interview(msg=end_interview_message)

    
    #@function_tool
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
                self.session.userdata.user_last_typed = time.time()
                code = parts[1].strip()
                if code:
                    final_marker = "<--final code-->"
                    if code.strip().endswith(final_marker):
                        code = code.rstrip().removesuffix(final_marker).rstrip()
                        if code:
                            logger.info(f"💬 Code: {code}")
                            # await self.session.say("Thank you for the submission.")
                            code_submission_prompt = "Candidate has submitted the code. Evaluate the code correctness and ask further question on it."
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
                        logger.info(f"💬 streaming code: {code}")
            else:
                await self.session.say("Please do not delete any existing code or text from the editor.")
