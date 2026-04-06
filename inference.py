"""
inference.py — Autonomous Inbox OS
Baseline inference script using OpenAI client.
Reads credentials from environment variables.
Runs all 3 tasks and produces reproducible scores.

Usage:
    export API_BASE_URL=https://api.openai.com/v1
    export MODEL_NAME=gpt-4o-mini
    export HF_TOKEN=your_token_here
    python inference.py
"""

import os
import json
import time
import sys
from openai import OpenAI

# ── Config ───────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
MAX_TOKENS   = 512
TEMPERATURE  = 0.0

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

SYSTEM_PROMPT = """You are an AI executive assistant managing an inbox.
For each email you receive, you must respond with a JSON object containing:
- action_type: one of [classify, reply, escalate, archive, schedule, ignore]
- classification: one of [urgent, important, low_priority, spam]
- reply_text: (optional) a brief reply if action_type is "reply"
- escalate_to: (optional) who to escalate to if action_type is "escalate"
- reasoning: a brief explanation of your decision (1-2 sentences)

Rules:
- urgent emails MUST be escalated or replied to immediately
- spam MUST be archived
- meeting requests should be scheduled or replied to
- important business emails should get a reply
- newsletters/promotions should be archived

Respond ONLY with valid JSON. No extra text."""


def call_agent(observation: dict) -> dict:
    """Call the LLM agent with the current email observation."""
    user_content = f"""
Inbox status:
- Emails remaining: {observation.get('inbox_size', '?')}
- Pending urgent: {observation.get('pending_urgent', 0)}
- Stress level: {observation.get('user_stress_level', 0):.0%}
- Time remaining: {observation.get('time_remaining', '?')} steps

Current email:
From: {observation.get('sender', '')}
Subject: {observation.get('subject', '')}
Body: {observation.get('body', '')}
Has attachment: {observation.get('has_attachment', False)}
Meeting request: {observation.get('meeting_request', False)}

What action do you take?"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = response.choices[0].message.content or "{}"
        text = text.strip().strip("```json").strip("```").strip()
        return json.loads(text)
    except Exception as e:
        return {
            "action_type": "archive",
            "classification": "low_priority",
            "reasoning": f"Fallback due to API error: {str(e)}"
        }


def run_task(task_id: int) -> dict:
    """Run a full episode for a given task and return grader score."""
    from server.environment import EmailEnvironment
    from models import EmailAction, ActionType, Priority

    def safe_enum(cls, val, default):
        try:
            return cls(val)
        except Exception:
            return default

    task_names = {1: "Spam Detection", 2: "Inbox Prioritization", 3: "Executive Decision-Making Under Pressure"}
    difficulties = {1: "easy", 2: "medium", 3: "hard"}

    env = EmailEnvironment(task_id=task_id)
    obs = env.reset()
    obs_dict = obs.model_dump()

    # [START] log — required format
    print(json.dumps({
        "event": "START",
        "task_id": task_id,
        "task_name": task_names[task_id],
        "difficulty": difficulties[task_id],
        "model": MODEL_NAME,
        "inbox_size": obs_dict.get("inbox_size", 0),
    }))

    step = 0
    total_reward = 0.0

    while True:
        step += 1
        action_dict = call_agent(obs_dict)

        action = EmailAction(
            action_type=safe_enum(ActionType, action_dict.get("action_type"), ActionType.archive),
            classification=safe_enum(Priority, action_dict.get("classification"), Priority.low_priority),
            reply_text=action_dict.get("reply_text"),
            escalate_to=action_dict.get("escalate_to"),
            reasoning=action_dict.get("reasoning"),
        )

        result = env.step(action)
        reward = result.reward.total
        total_reward += reward

        # [STEP] log — required format
        print(json.dumps({
            "event": "STEP",
            "task_id": task_id,
            "step": step,
            "observation": {
                "email_id": obs_dict.get("email_id"),
                "subject": obs_dict.get("subject", "")[:60],
                "sender": obs_dict.get("sender", ""),
            },
            "action": {
                "action_type": action.action_type.value,
                "classification": action.classification.value if action.classification else None,
                "reasoning": action.reasoning,
            },
            "reward": round(reward, 4),
            "done": result.done,
            "stress": round(env._stress, 3),
        }))

        if result.done:
            break
        obs_dict = result.observation.model_dump()
        time.sleep(0.05)

    grader = env.run_grader()

    # [END] log — required format
    print(json.dumps({
        "event": "END",
        "task_id": task_id,
        "task_name": task_names[task_id],
        "difficulty": difficulties[task_id],
        "steps": step,
        "total_reward": round(total_reward, 4),
        "score": grader["score"],
        "label_accuracy": grader["details"]["label_accuracy"],
        "action_accuracy": grader["details"]["action_accuracy"],
        "missed_urgent": grader["details"]["missed_urgent"],
        "final_stress": grader["details"]["final_stress"],
    }))

    return {
        "task_id": task_id,
        "score": grader["score"],
        "details": grader["details"],
        "steps": step,
        "total_reward": round(total_reward, 4),
    }


def main():
    import server.environment as env_module

    results = []
    for task_id in [1, 2, 3]:
        result = run_task(task_id)
        results.append(result)

    avg_score = sum(r["score"] for r in results) / len(results)

    # Final summary
    print(json.dumps({
        "event": "SUMMARY",
        "model": MODEL_NAME,
        "average_score": round(avg_score, 4),
        "results": results,
    }))

    # Save to file
    with open("baseline_scores.json", "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "results": results,
            "average_score": round(avg_score, 4),
        }, f, indent=2)


if __name__ == "__main__":
    main()
