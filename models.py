from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union
from enum import Enum


class ActionType(str, Enum):
    classify = "classify"
    reply = "reply"
    escalate = "escalate"
    archive = "archive"
    schedule = "schedule"
    ignore = "ignore"


class Priority(str, Enum):
    urgent = "urgent"
    important = "important"
    low_priority = "low_priority"
    spam = "spam"


class EmailObservation(BaseModel):
    email_id: int
    subject: str
    body: str
    sender: str
    has_attachment: bool = False
    meeting_request: bool = False
    inbox_size: int = 0
    pending_urgent: int = 0
    pending_important: int = 0
    user_stress_level: float = Field(ge=0.0, le=1.0, default=0.0)
    time_remaining: int = 100
    step_count: int = 0


class EmailAction(BaseModel):
    action_type: ActionType
    classification: Optional[Priority] = Field(default=None)
    reply_text: Optional[str] = Field(default=None)
    escalate_to: Optional[str] = Field(default=None)
    scheduled_time: Optional[str] = Field(default=None)
    reasoning: Optional[str] = Field(default=None)


class EmailReward(BaseModel):
    total: float
    classification_score: float = 0.0
    action_score: float = 0.0
    time_penalty: float = 0.0
    stress_penalty: float = 0.0
    missed_urgent_penalty: float = 0.0
    reply_quality_score: float = 0.0


class EmailState(BaseModel):
    episode_id: str
    step_count: int
    current_index: int
    inbox_size: int
    processed: int
    correct_classifications: int
    missed_urgent: int
    user_stress_level: float
    time_remaining: int
    cumulative_reward: float
    done: bool = False


class TaskDefinition(BaseModel):
    task_id: int
    name: str
    description: str
    difficulty: str
    email_filter: Optional[str] = Field(default=None)


class TaskResult(BaseModel):
    task_id: int
    score: float
    details: Dict[str, Any] = {}


class StepResult(BaseModel):
    observation: Optional[EmailObservation] = Field(default=None)
    reward: EmailReward
    done: bool
    info: Dict[str, Any] = {}
