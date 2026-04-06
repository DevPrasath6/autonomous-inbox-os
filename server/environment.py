import json
import uuid
import os
import sys

# Allow imports from parent dir when running as server
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    EmailObservation, EmailAction, EmailState, EmailReward,
    StepResult, TaskDefinition, TaskResult, ActionType, Priority
)
from typing import List, Dict, Any


TASKS = [
    TaskDefinition(
        task_id=1,
        name="Spam Detection",
        description="Identify and archive spam emails from the inbox. Agent must classify each email and archive spam.",
        difficulty="easy",
        email_filter="spam"
    ),
    TaskDefinition(
        task_id=2,
        name="Inbox Prioritization",
        description="Classify all emails by priority (urgent/important/low_priority/spam) and take the correct action for each.",
        difficulty="medium",
        email_filter=None
    ),
    TaskDefinition(
        task_id=3,
        name="Executive Decision-Making Under Pressure",
        description="Handle a flooded inbox with conflicting priorities, urgent deadlines, and time pressure. Agent must classify, reply, escalate, and schedule while managing stress levels.",
        difficulty="hard",
        email_filter=None
    ),
]


class EmailEnvironment:
    """
    Autonomous Inbox OS — OpenEnv-compliant environment.
    Simulates a real-world executive inbox where an AI agent
    must manage emails under time and cognitive pressure.
    """

    def __init__(self, task_id: int = 2, flood_mode: bool = False):
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "emails.json")
        with open(data_path) as f:
            self._all_emails = json.load(f)

        self.task_id = task_id
        self.flood_mode = flood_mode
        self._episode_id = str(uuid.uuid4())
        self._emails = self._filter_emails_for_task(task_id)
        self._index = 0
        self._step_count = 0
        self._stress = 0.0
        self._cumulative_reward = 0.0
        self._correct = 0
        self._missed_urgent = 0
        self._pending_urgent = sum(1 for e in self._emails if e["label"] == "urgent")
        self._pending_important = sum(1 for e in self._emails if e["label"] == "important")
        self._time_remaining = len(self._emails) * 3  # 3 steps per email budget
        self._predictions: List[Dict] = []
        self._done = False

    def _filter_emails_for_task(self, task_id: int) -> List[Dict]:
        if task_id == 1:
            # Easy: spam + important mix
            return [e for e in self._all_emails if e["category"] in ("spam", "hr", "social", "newsletter", "transactional")]
        elif task_id == 2:
            # Medium: all categories, full classification
            return self._all_emails[:]
        else:
            # Hard: all, flood mode — injects extra urgents
            base = self._all_emails[:]
            extra = [e for e in self._all_emails if e["label"] == "urgent"]
            return base + extra  # doubled urgents = more pressure

    def reset(self) -> EmailObservation:
        self._episode_id = str(uuid.uuid4())
        self._emails = self._filter_emails_for_task(self.task_id)
        self._index = 0
        self._step_count = 0
        self._stress = 0.0
        self._cumulative_reward = 0.0
        self._correct = 0
        self._missed_urgent = 0
        self._pending_urgent = sum(1 for e in self._emails if e["label"] == "urgent")
        self._pending_important = sum(1 for e in self._emails if e["label"] == "important")
        self._time_remaining = len(self._emails) * 3
        self._predictions = []
        self._done = False
        return self._build_observation()

    def step(self, action: EmailAction) -> StepResult:
        if self._done:
            raise ValueError("Episode is done. Call reset() to start a new episode.")

        email = self._emails[self._index]
        reward = self._compute_reward(email, action)

        # Update stress engine
        self._update_stress(email, action)

        # Track prediction
        self._predictions.append({
            "email_id": email["id"],
            "expected_label": email["label"],
            "predicted_label": action.classification.value if action.classification else None,
            "expected_action": email["expected_action"],
            "taken_action": action.action_type.value,
            "reward": reward.total,
        })

        if action.classification and action.classification.value == email["label"]:
            self._correct += 1

        self._index += 1
        self._step_count += 1
        self._time_remaining -= 1
        self._cumulative_reward += reward.total

        # Update pending counts
        if email["label"] == "urgent":
            self._pending_urgent = max(0, self._pending_urgent - 1)
        if email["label"] == "important":
            self._pending_important = max(0, self._pending_important - 1)

        done = self._index >= len(self._emails) or self._time_remaining <= 0
        self._done = done

        obs = None if done else self._build_observation()
        return StepResult(observation=obs, reward=reward, done=done, info={
            "email_id": email["id"],
            "correct_action": email["expected_action"],
            "stress": self._stress,
        })

    def _compute_reward(self, email: Dict, action: EmailAction) -> EmailReward:
        classification_score = 0.0
        action_score = 0.0
        time_penalty = -0.05  # always costs time
        stress_penalty = -self._stress * 0.2
        missed_urgent_penalty = 0.0
        reply_quality_score = 0.0

        # Classification scoring
        if action.classification:
            if action.classification.value == email["label"]:
                classification_score = 1.0
            elif self._is_adjacent_label(action.classification.value, email["label"]):
                classification_score = 0.4  # partial credit
            else:
                classification_score = -0.3

        # Action scoring
        correct_action = email["expected_action"]
        taken_action = action.action_type.value

        if taken_action == correct_action:
            action_score = 1.0
        elif self._is_compatible_action(taken_action, correct_action):
            action_score = 0.5
        else:
            action_score = -0.3

        # Missed urgent penalty
        if email["label"] == "urgent" and taken_action == "archive":
            missed_urgent_penalty = -2.0
            self._missed_urgent += 1

        # Reply quality (if reply provided)
        if action.action_type == ActionType.reply and action.reply_text:
            reply_quality_score = self._score_reply(action.reply_text, email)

        # Reasoning bonus (chain-of-thought)
        reasoning_bonus = 0.1 if action.reasoning and len(action.reasoning) > 20 else 0.0

        total = (
            classification_score * 0.35
            + action_score * 0.35
            + time_penalty
            + stress_penalty
            + missed_urgent_penalty
            + reply_quality_score * 0.2
            + reasoning_bonus
        )

        return EmailReward(
            total=round(total, 4),
            classification_score=classification_score,
            action_score=action_score,
            time_penalty=time_penalty,
            stress_penalty=round(stress_penalty, 4),
            missed_urgent_penalty=missed_urgent_penalty,
            reply_quality_score=reply_quality_score,
        )

    def _update_stress(self, email: Dict, action: EmailAction):
        # Stress increases with unhandled urgents
        if self._pending_urgent > 3:
            self._stress = min(1.0, self._stress + 0.08)

        # Stress increases with wrong decisions on urgent
        if email["label"] == "urgent" and action.action_type.value != "escalate":
            self._stress = min(1.0, self._stress + 0.15)

        # Correct decisions reduce stress
        if action.action_type.value == email["expected_action"]:
            self._stress = max(0.0, self._stress - 0.05)

        # Time pressure increases stress in hard mode
        if self.task_id == 3 and self._time_remaining < 10:
            self._stress = min(1.0, self._stress + 0.1)

    def _is_adjacent_label(self, predicted: str, actual: str) -> bool:
        adjacency = {
            ("urgent", "important"), ("important", "urgent"),
            ("low_priority", "spam"), ("spam", "low_priority"),
        }
        return (predicted, actual) in adjacency

    def _is_compatible_action(self, taken: str, correct: str) -> bool:
        compatible = {
            ("reply", "escalate"), ("escalate", "reply"),
            ("archive", "ignore"), ("ignore", "archive"),
        }
        return (taken, correct) in compatible

    def _score_reply(self, reply_text: str, email: Dict) -> float:
        score = 0.0
        reply_lower = reply_text.lower()
        # Check relevance signals
        if any(word in reply_lower for word in ["thank", "acknowledge", "received"]):
            score += 0.3
        if len(reply_text) > 50:
            score += 0.3
        if email["category"] == "meeting" and any(w in reply_lower for w in ["schedule", "available", "confirm", "time"]):
            score += 0.4
        if email["category"] == "client" and any(w in reply_lower for w in ["proposal", "discuss", "call", "forward"]):
            score += 0.4
        return min(1.0, score)

    def _build_observation(self) -> EmailObservation:
        if self._index >= len(self._emails):
            return None
        email = self._emails[self._index]
        return EmailObservation(
            email_id=email["id"],
            subject=email["subject"],
            body=email["body"],
            sender=email["sender"],
            has_attachment=email.get("has_attachment", False),
            meeting_request=email.get("meeting_request", False),
            inbox_size=len(self._emails) - self._index,
            pending_urgent=self._pending_urgent,
            pending_important=self._pending_important,
            user_stress_level=round(self._stress, 3),
            time_remaining=self._time_remaining,
            step_count=self._step_count,
        )

    def state(self) -> EmailState:
        return EmailState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            current_index=self._index,
            inbox_size=len(self._emails),
            processed=self._index,
            correct_classifications=self._correct,
            missed_urgent=self._missed_urgent,
            user_stress_level=round(self._stress, 3),
            time_remaining=self._time_remaining,
            cumulative_reward=round(self._cumulative_reward, 4),
            done=self._done,
        )

    def get_tasks(self) -> List[TaskDefinition]:
        return TASKS

    def run_grader(self) -> Dict[str, Any]:
        """Run the programmatic grader for completed episode."""
        if not self._predictions:
            return {"score": 0.0, "details": {}}

        n = len(self._predictions)
        correct_labels = sum(1 for p in self._predictions if p["expected_label"] == p["predicted_label"])
        correct_actions = sum(1 for p in self._predictions if p["expected_action"] == p["taken_action"])
        missed_urgent = self._missed_urgent
        avg_reward = self._cumulative_reward / n if n > 0 else 0.0

        label_accuracy = correct_labels / n
        action_accuracy = correct_actions / n
        urgent_penalty = min(1.0, missed_urgent * 0.25)
        stress_factor = 1.0 - self._stress * 0.3

        # Weighted final score 0.0–1.0
        raw = (
            label_accuracy * 0.35
            + action_accuracy * 0.35
            + stress_factor * 0.2
            + min(1.0, (avg_reward + 1.0) / 2.0) * 0.1
        ) - urgent_penalty * 0.2

        score = max(0.0, min(1.0, raw))

        return {
            "score": round(score, 4),
            "details": {
                "emails_processed": n,
                "label_accuracy": round(label_accuracy, 4),
                "action_accuracy": round(action_accuracy, 4),
                "missed_urgent": missed_urgent,
                "final_stress": round(self._stress, 3),
                "cumulative_reward": round(self._cumulative_reward, 4),
                "avg_reward_per_step": round(avg_reward, 4),
            }
        }
