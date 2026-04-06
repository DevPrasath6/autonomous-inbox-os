"""
inference.py — Autonomous Inbox OS
Baseline inference script using OpenAI client.

MANDATORY ENV VARS:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

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
from typing import List, Optional
from openai import OpenAI

# ── Config ───────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN", "")
MAX_TOKENS   = 512
TEMPERATURE  = 0.0
BENCHMARK    = "autonomous_inbox_os"

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

SYSTEM_PROMPT = """You are an AI executive assistant managing an inbox.
For each email you receive, respond with a JSON object:
- action_type: one of [classify, reply, escalate, archive, schedule, ignore]
- classification: one of [urgent, important, low_priority, spam]
- reply_text: (optional) brief reply if action_type is reply
- escalate_to: (optional) who to escalate to
- reasoning: brief explanation (1-2 sentences)

Rules:
- urgent emails MUST be escalated immediately
- spam MUST be archived
- meeting requests should be scheduled or replied to
- important business emails should get a reply
- newsletters/promotions should be archived

Respond ONLY with valid JSON. No extra text."""


# ── Logging helpers (MANDATORY FORMAT) ───────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ── Agent ─────────────────────────────────────────────────────────────────────
def call_agent(observation: dict) -> dict:
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
            "reasoning": f"Fallback: {str(e)}"
        }


# ── Task runner ───────────────────────────────────────────────────────────────
def run_task(task_id: int) -> dict:
    from server.environment import EmailEnvironment
    from models import EmailAction, ActionType, Priority

    task_names = {
        1: "spam-detection",
        2: "inbox-prioritization",
        3: "executive-decision-making"
    }
    task_name = task_names[task_id]

    def safe_enum(cls, val, default):
        try:
            return cls(val)
        except Exception:
            return default

    env = EmailEnvironment(task_id=task_id)
    obs = env.reset()
    obs_dict = obs.model_dump()

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, 100):
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
            done = result.done
            error = None

            # action string for log
            action_str = f"{action.action_type.value}({action.classification.value if action.classification else 'none'})"

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            rewards.append(reward)
            steps_taken = step

            if done:
                break

            obs_dict = result.observation.model_dump()
            time.sleep(0.05)

        grader = env.run_grader()
        score = grader["score"]
        success = score >= 0.1

    except Exception as e:
        error_msg = str(e)
        log_step(step=steps_taken + 1, action="error", reward=0.0, done=True, error=error_msg)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "task_name": task_name,
        "score": score,
        "steps": steps_taken,
        "total_reward": round(sum(rewards), 4),
        "rewards": rewards,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    results = []
    for task_id in [1, 2, 3]:
        result = run_task(task_id)
        results.append(result)

    avg_score = sum(r["score"] for r in results) / len(results)

    # Save baseline scores
    with open("baseline_scores.json", "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "average_score": round(avg_score, 4),
            "results": results,
        }, f, indent=2)


if __name__ == "__main__":
    main()
