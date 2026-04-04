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

# ── Config ──────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
MAX_TOKENS   = 512
TEMPERATURE  = 0.0  # deterministic for reproducibility

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
        # Strip markdown fences if present
        text = text.strip().strip("```json").strip("```").strip()
        return json.loads(text)
    except Exception as e:
        print(f"  [Agent error] {e} — using fallback action")
        return {
            "action_type": "archive",
            "classification": "low_priority",
            "reasoning": "Fallback due to API error."
        }


def run_task(task_id: int, env_module) -> dict:
    """Run a full episode for a given task and return grader score."""
    from server.environment import EmailEnvironment

    print(f"\n{'='*60}")
    print(f"  TASK {task_id}: {['', 'Spam Detection', 'Inbox Prioritization', 'Executive Decision-Making'][task_id]}")
    print(f"  Difficulty: {['', 'Easy', 'Medium', 'Hard'][task_id]}")
    print(f"{'='*60}")

    env = EmailEnvironment(task_id=task_id)
    obs = env.reset()
    obs_dict = obs.model_dump()

    step = 0
    total_reward = 0.0

    while True:
        step += 1
        print(f"  Step {step:02d} | 📧 {obs_dict['subject'][:50]}...", end="")

        action_dict = call_agent(obs_dict)

        # Build action from agent response
        from models import EmailAction, ActionType, Priority

        def safe_enum(cls, val, default):
            try:
                return cls(val)
            except Exception:
                return default

        action = EmailAction(
            action_type=safe_enum(ActionType, action_dict.get("action_type"), ActionType.archive),
            classification=safe_enum(Priority, action_dict.get("classification"), Priority.low_priority),
            reply_text=action_dict.get("reply_text"),
            escalate_to=action_dict.get("escalate_to"),
            reasoning=action_dict.get("reasoning"),
        )

        result = env.step(action)
        total_reward += result.reward.total

        print(f" → {action.action_type.value} | reward: {result.reward.total:+.3f} | stress: {env._stress:.0%}")

        if result.done:
            break
        obs_dict = result.observation.model_dump()

        time.sleep(0.1)  # rate limit safety

    grader = env.run_grader()
    state = env.state()

    print(f"\n  ✅ Task {task_id} complete")
    print(f"     Score:           {grader['score']:.4f}")
    print(f"     Label accuracy:  {grader['details']['label_accuracy']:.1%}")
    print(f"     Action accuracy: {grader['details']['action_accuracy']:.1%}")
    print(f"     Missed urgent:   {grader['details']['missed_urgent']}")
    print(f"     Final stress:    {grader['details']['final_stress']:.0%}")
    print(f"     Total reward:    {total_reward:+.4f}")

    return {
        "task_id": task_id,
        "score": grader["score"],
        "details": grader["details"],
        "steps": step,
        "total_reward": round(total_reward, 4),
    }


def main():
    print("\n" + "🚀 " * 20)
    print("  AUTONOMOUS INBOX OS — Baseline Inference")
    print(f"  Model:    {MODEL_NAME}")
    print(f"  Base URL: {API_BASE_URL}")
    print("🚀 " * 20)

    import server.environment as env_module

    results = []
    for task_id in [1, 2, 3]:
        result = run_task(task_id, env_module)
        results.append(result)

    print("\n" + "=" * 60)
    print("  FINAL SCORES")
    print("=" * 60)
    for r in results:
        bar_len = int(r["score"] * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        difficulty = ["", "Easy  ", "Medium", "Hard  "][r["task_id"]]
        print(f"  Task {r['task_id']} [{difficulty}] [{bar}] {r['score']:.4f}")

    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"\n  Average score: {avg_score:.4f}")
    print("=" * 60)

    # Write results to file for reproducibility
    with open("baseline_scores.json", "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "results": results,
            "average_score": round(avg_score, 4),
        }, f, indent=2)
    print("\n  📄 Results saved to baseline_scores.json")


if __name__ == "__main__":
    main()
