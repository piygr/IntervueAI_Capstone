# modules/memory.py

import logging
import json
import os
import time
from typing import List, Optional
from pydantic import BaseModel
from pydantic.json import pydantic_encoder

logger = logging.getLogger(__name__)

class InterviewQuestionItem(BaseModel):
    question: str
    evaluation_category: str
    evaluation_depth: str

class InterviewPlan(BaseModel):
    scheduled_duration_in_minutes: int
    questions: List[InterviewQuestionItem]

class FollowupItem(BaseModel):
    agent: str
    type: str
    timestamp: float
    candidate: Optional["CandidateResponseItem"] = {}


class CandidateResponseItem(BaseModel):
    response: str
    timestamp: float
    followup: Optional[FollowupItem] = None


class ConversationItem(BaseModel):
    type: str
    question_index: int
    time_spent_on_question_in_minutes: int
    agent: str
    timestamp: float
    question_status: str
    question_remarks: str
    candidate: Optional[CandidateResponseItem] = {}

'''
{
    "session_id": "",
    "interview_plan": {
        "scheduled_duration_in_minutes": 45,
        "questions": [
            {
                "question": "",
                "evaluation_category": "",
                "evaluation_depth": ""
            },
            ...
        ]
    },
    "conversation": [
        {
            "type": "question", #"greet"
            "question_index": 0,
            "time_spent_on_question_in_minutes": 0
            "agent": "What is react",
            "timestamp": 12345,
            "question_status": "ongoing" #done, "not_completed"
            "question_remarks": "skipped_by_candidate", #not enough time to complete, agent skipped (In case of not_completed question_status)
            "candidate": {
                "response": "",
                "timestamp": ""
                "followup": {
                    "agent": "",
                    "type": "hint" #"probe", "moved_next_question",
                    "timestamp": 12349,
                    "candidate": {
                        "response": "",
                        "timestamp": 12456,
                        "followup": {
                            ...
                        }
                    }
                }
            }
        },
        ...
    ]
}
'''

class MemoryManager:
    """Manages session memory (read/write/append)."""

    def __init__(self, session_id: str, memory_dir: str = "logs/memory", interview_plan: InterviewPlan = None):
        self.session_id = session_id
        self.memory_dir = memory_dir
        self.memory_path = os.path.join(memory_dir, f'{session_id}.json')
        self.conversation: List[ConversationItem] = []
        self.interview_plan: InterviewPlan = interview_plan
        
        if not os.path.exists(self.memory_dir):
            os.makedirs(self.memory_dir)

        self.load()

    def load(self):
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
                self.session_id = raw.get("session_id")
                self.interview_plan = InterviewPlan(**raw.get("interview_plan"))
                self.conversation = List(ConversationItem(**item) for item in raw.get("conversation"))

    def save(self):
        # Before opening the file for writing
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        with open(self.memory_path, "w", encoding="utf-8") as f:
            raw = dict(
                session_id=self.session_id,
                interview_plan=self.interview_plan,
                conversation=self.conversation
            )
            
            json.dump(raw, f, indent=2, default=pydantic_encoder)

    def to_json(self):
        raw = dict(
                session_id=self.session_id,
                interview_plan=self.interview_plan.model_dump_json(),
                conversation=[item.model_dump_json() for item in self.conversation]
            )
        return json.loads(json.dumps(raw))
        
    def __str__(self) -> str:
        return json.dumps(self.to_json())

    def add_conversation_item(self, item: ConversationItem):
        self.conversation.append(item)
        self.save()

    def add_agent_question(self,
                           question_index: int,
                           agent: str,
                           timestamp: float):
        
        item = ConversationItem(
            type="question",
            question_index=question_index,
            time_spent_on_question_in_minutes=0,
            agent=agent,
            timestamp=timestamp,
            question_status="ongoing",
            question_remarks="",
            candidate=None
        )
        self.conversation.append(item)
        self.save()
        return item

    def update_candidate_response(self,
                                  question_index: int,
                                  response: str,
                                  timestamp: float):
        
        def _recursive_candidate_response(item: CandidateResponseItem):
            if item.followup:
                if item.followup.candidate:
                    return _recursive_candidate_response(item.followup.candidate)
                else:
                    new_candidate_response = CandidateResponseItem(
                        response=response,
                        timestamp=timestamp
                    )
                    item.followup.candidate = new_candidate_response
                    return new_candidate_response
            else:
                if not item.response:
                    item.response = response
                    item.timestamp = timestamp
                else:
                    item.response = item.response + "\n" + response
                    item.timestamp = timestamp
                
                return item

        candidate_response = None
        for item in self.conversation:
            if item.question_index == question_index:
                
                item.time_spent_on_question_in_minutes = int( (time.time() - item.timestamp) / 60 )
                
                if item.candidate:
                    candidate_response = _recursive_candidate_response(item.candidate)
                else:
                    new_candidate_response = CandidateResponseItem(
                        response=response,
                        timestamp=timestamp
                    )
                    item.candidate = new_candidate_response
                    candidate_response = new_candidate_response
        
        self.save()
        return candidate_response

    def add_followup_item(self, 
                          question_index: str,
                          followup: str,
                          followup_type: str,
                          timestamp: float):
        
        def _recursive_followup(item: FollowupItem):
            if item.candidate:
                if item.candidate.followup:
                    _recursive_followup(item.candidate.followup)
                else:
                    new_followup = FollowupItem(
                            agent=followup,
                            type=followup_type,
                            timestamp=timestamp
                        )
                    item.candidate.followup = new_followup
            else:
                new_candidate_response = CandidateResponseItem(
                        response=None,
                        timestamp=None,
                        followup=FollowupItem(
                            agent=followup,
                            type=followup_type,
                            timestamp=timestamp
                        )
                    )
                item.candidate = new_candidate_response


        for item in self.conversation:
            if item.question_index == question_index:

                item.time_spent_on_question_in_minutes = int( (time.time() - item.timestamp) / 60 )
                
                if item.candidate:
                    if item.candidate.followup:
                        _recursive_followup(item.candidate.followup)
                    else:
                        new_followup = FollowupItem(
                            agent=followup,
                            type=followup_type,
                            timestamp=timestamp
                        )
                        item.candidate.followup = new_followup
                else:
                    
                    new_candidate_response = CandidateResponseItem(
                        response=None,
                        timestamp=None,
                        followup=FollowupItem(
                            agent=followup,
                            type=followup_type,
                            timestamp=timestamp
                        )
                    )
                    item.candidate = new_candidate_response
        
        self.save()

    '''
    def add_tool_call(
        self, tool_name: str, tool_args: dict, tags: Optional[List[str]] = None
    ):
        item = ConversationItem(
            timestamp=time.time(),
            type="tool_call",
            text=f"Called {tool_name} with {tool_args}",
            tool_name=tool_name,
            tool_args=tool_args,
            tags=tags or [],
        )
        self.add(item)

    def add_tool_output(
        self, tool_name: str, tool_args: dict, tool_result: dict, success: bool, tags: Optional[List[str]] = None
    ):
        item = ConversationItem(
            timestamp=time.time(),
            type="tool_output",
            text=f"Output of {tool_name}: {tool_result}",
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
            success=success,  # 🆕 Track success!
            tags=tags or [],
        )
        self.add(item)

    def add_final_answer(self, text: str):
        item = ConversationItem(
            timestamp=time.time(),
            type="final_answer",
            text=text,
            final_answer=text,
        )
        self.add(item)

    def find_recent_successes(self, limit: int = 5) -> List[str]:
        """Find tool names which succeeded recently."""
        tool_successes = []

        # Search from newest to oldest
        for item in reversed(self.items):
            if item.type == "tool_output" and item.success:
                if item.tool_name and item.tool_name not in tool_successes:
                    tool_successes.append(item.tool_name)
            if len(tool_successes) >= limit:
                break

        return tool_successes

    def add_tool_success(self, tool_name: str, success: bool):
        """Patch last tool call or output for a given tool with success=True/False."""

        # Search backwards for latest matching tool call/output
        for item in reversed(self.items):
            if item.tool_name == tool_name and item.type in {"tool_call", "tool_output"}:
                item.success = success
                log("memory", f"✅ Marked {tool_name} as success={success}")
                self.save()
                return

        log("memory", f"⚠️ Tried to mark {tool_name} as success={success} but no matching memory found.")

    def get_session_items(self) -> List[MemoryItem]:
        """
        Return all memory items for current session.
        """
        return self.items
'''