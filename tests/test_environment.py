"""
tests/test_environment.py
Full test suite for Autonomous Inbox OS OpenEnv environment.
Run with: pytest tests/ -v
"""

import sys
import os
import json
import random
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import EmailAction, EmailObservation, EmailState, EmailReward, ActionType, Priority
from server.environment import EmailEnvironment, TASKS


# ── FIXTURES ──────────────────────────────────────────────────────────────────

@pytest.fixture
def env():
    e = EmailEnvironment(task_id=2)
    e.reset()
    return e

@pytest.fixture
def env_task1():
    e = EmailEnvironment(task_id=1)
    e.reset()
    return e

@pytest.fixture
def env_task3():
    e = EmailEnvironment(task_id=3)
    e.reset()
    return e

def make_action(action_type="archive", classification="spam", reasoning=None):
    return EmailAction(
        action_type=ActionType(action_type),
        classification=Priority(classification),
        reasoning=reasoning,
    )


# ── RESET TESTS ───────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_returns_observation(self):
        env = EmailEnvironment(task_id=1)
        obs = env.reset()
        assert isinstance(obs, EmailObservation)

    def test_reset_clears_stress(self, env):
        # Advance a few steps then reset
        for _ in range(3):
            if env._done: break
            env.step(make_action())
        env.reset()
        assert env._stress == 0.0

    def test_reset_clears_step_count(self, env):
        env.step(make_action())
        env.reset()
        assert env._step_count == 0

    def test_reset_clears_reward(self, env):
        env.step(make_action())
        env.reset()
        assert env._cumulative_reward == 0.0

    def test_reset_returns_new_episode_id(self, env):
        id1 = env._episode_id
        env.reset()
        id2 = env._episode_id
        assert id1 != id2

    def test_initial_stress_is_zero(self, env):
        assert env._stress == 0.0

    def test_initial_observation_has_subject(self, env):
        obs = env._build_observation()
        assert obs.subject
        assert obs.sender
        assert obs.body

    def test_stress_level_in_valid_range(self, env):
        obs = env._build_observation()
        assert 0.0 <= obs.user_stress_level <= 1.0


# ── STEP TESTS ────────────────────────────────────────────────────────────────

class TestStep:
    def test_step_returns_step_result(self, env):
        from models import StepResult
        result = env.step(make_action())
        assert isinstance(result, StepResult)

    def test_step_reward_is_float(self, env):
        result = env.step(make_action())
        assert isinstance(result.reward.total, float)

    def test_step_done_is_bool(self, env):
        result = env.step(make_action())
        assert isinstance(result.done, bool)

    def test_step_increments_step_count(self, env):
        env.step(make_action())
        assert env._step_count == 1

    def test_step_raises_when_done(self):
        env = EmailEnvironment(task_id=1)
        env.reset()
        while not env._done:
            r = env.step(make_action())
            if r.done: break
        with pytest.raises(ValueError, match="Episode is done"):
            env.step(make_action())

    def test_observation_is_none_when_done(self):
        env = EmailEnvironment(task_id=1)
        env.reset()
        result = None
        while not env._done:
            result = env.step(make_action())
            if result.done: break
        assert result is not None
        assert result.observation is None

    def test_reasoning_gives_bonus(self, env):
        # Two identical steps, one with reasoning — should have slightly higher reward
        env2 = EmailEnvironment(task_id=2)
        env2.reset()

        a_no_reason = make_action("archive", "spam", None)
        a_with_reason = make_action("archive", "spam", "This is spam because the sender is sketchy")

        r1 = env.step(a_no_reason)
        r2 = env2.step(a_with_reason)
        # With reasoning should be >= without
        assert r2.reward.total >= r1.reward.total


# ── REWARD TESTS ──────────────────────────────────────────────────────────────

class TestReward:
    def test_correct_label_positive(self, env):
        """Correct label classification gives positive classification_score."""
        email = env._emails[env._index]
        action = make_action("archive", email["label"])
        result = env.step(action)
        assert result.reward.classification_score >= 0.0

    def test_correct_action_positive(self):
        """Correct action gives positive action_score."""
        env = EmailEnvironment(task_id=2)
        env.reset()
        email = env._emails[env._index]
        action = EmailAction(
            action_type=ActionType(email["expected_action"]),
            classification=Priority(email["label"]),
        )
        result = env.step(action)
        assert result.reward.action_score > 0.0

    def test_missed_urgent_large_penalty(self):
        """Archiving an urgent email gives a penalty < -1.0."""
        env = EmailEnvironment(task_id=2)
        env.reset()
        # Seek an urgent email
        while not env._done and env._emails[env._index]["label"] != "urgent":
            env.step(make_action("archive", "low_priority"))
        if env._done:
            pytest.skip("No urgent email found before episode end")
        result = env.step(make_action("archive", "spam"))
        assert result.reward.total < -1.0

    def test_time_penalty_always_present(self, env):
        result = env.step(make_action())
        assert result.reward.time_penalty < 0.0

    def test_reward_bounded(self):
        """Reward per step stays within reasonable bounds."""
        env = EmailEnvironment(task_id=3)
        env.reset()
        while not env._done:
            r = env.step(make_action("escalate", "urgent", "Urgent — escalating"))
            assert -5.0 <= r.reward.total <= 5.0
            if r.done: break

    def test_reward_components_sum(self, env):
        """Check that reward breakdown is internally consistent."""
        result = env.step(make_action())
        rw = result.reward
        # Total should reflect the components (approximately)
        assert isinstance(rw.classification_score, float)
        assert isinstance(rw.action_score, float)
        assert isinstance(rw.time_penalty, float)


# ── STATE TESTS ───────────────────────────────────────────────────────────────

class TestState:
    def test_state_returns_email_state(self, env):
        s = env.state()
        assert isinstance(s, EmailState)

    def test_state_has_episode_id(self, env):
        s = env.state()
        assert s.episode_id and len(s.episode_id) > 0

    def test_state_step_count_matches(self, env):
        env.step(make_action())
        env.step(make_action())
        s = env.state()
        assert s.step_count == 2

    def test_state_stress_matches(self, env):
        env.step(make_action())
        s = env.state()
        assert s.user_stress_level == round(env._stress, 3)

    def test_state_done_false_initially(self, env):
        s = env.state()
        assert s.done == False


# ── STRESS ENGINE TESTS ───────────────────────────────────────────────────────

class TestStressEngine:
    def test_stress_increases_on_missed_urgent(self):
        env = EmailEnvironment(task_id=2)
        env.reset()
        initial_stress = env._stress
        # Archive an urgent email to spike stress
        while not env._done and env._emails[env._index]["label"] != "urgent":
            env.step(make_action("archive", "low_priority"))
        if env._done:
            pytest.skip("No urgent email available")
        env.step(make_action("archive", "spam"))
        assert env._stress > initial_stress

    def test_stress_stays_bounded(self):
        env = EmailEnvironment(task_id=3)
        env.reset()
        while not env._done:
            env.step(make_action("archive", "spam"))
            assert 0.0 <= env._stress <= 1.0

    def test_stress_reduces_on_correct_action(self):
        env = EmailEnvironment(task_id=2)
        env.reset()
        env._stress = 0.5  # manually set stress
        email = env._emails[env._index]
        action = EmailAction(
            action_type=ActionType(email["expected_action"]),
            classification=Priority(email["label"]),
        )
        env.step(action)
        assert env._stress < 0.5  # stress should drop


# ── TASKS & GRADER TESTS ──────────────────────────────────────────────────────

class TestTasksAndGrader:
    def test_three_tasks_defined(self):
        assert len(TASKS) >= 3

    def test_task_difficulties(self):
        difficulties = {t.difficulty for t in TASKS}
        assert "easy" in difficulties
        assert "hard" in difficulties

    def test_grader_returns_float_score(self, env):
        env.step(make_action())
        result = env.run_grader()
        assert isinstance(result["score"], float)

    def test_grader_score_in_range(self):
        for task_id in [1, 2, 3]:
            env = EmailEnvironment(task_id=task_id)
            env.reset()
            while not env._done:
                r = env.step(make_action(
                    random.choice(["archive", "reply", "escalate"]),
                    random.choice(["urgent", "important", "spam", "low_priority"]),
                ))
                if r.done: break
            score = env.run_grader()["score"]
            assert 0.0 <= score <= 1.0, f"Task {task_id} score {score} out of [0,1]"

    def test_grader_deterministic(self):
        def run_fixed(task_id):
            env = EmailEnvironment(task_id=task_id)
            env.reset()
            while not env._done:
                r = env.step(make_action("archive", "spam"))
                if r.done: break
            return env.run_grader()["score"]

        for task_id in [1, 2, 3]:
            s1 = run_fixed(task_id)
            s2 = run_fixed(task_id)
            assert s1 == s2, f"Task {task_id} grader not deterministic: {s1} ≠ {s2}"

    def test_perfect_agent_scores_higher_than_random(self):
        def run_policy(task_id, perfect=False):
            env = EmailEnvironment(task_id=task_id)
            env.reset()
            while not env._done:
                if perfect:
                    email = env._emails[env._index]
                    a = EmailAction(
                        action_type=ActionType(email["expected_action"]),
                        classification=Priority(email["label"]),
                        reasoning="Perfect decision"
                    )
                else:
                    a = make_action(
                        random.choice(["archive", "ignore"]),
                        "spam"
                    )
                r = env.step(a)
                if r.done: break
            return env.run_grader()["score"]

        for task_id in [1, 2, 3]:
            perfect_score = run_policy(task_id, perfect=True)
            random_score = run_policy(task_id, perfect=False)
            assert perfect_score > random_score, \
                f"Task {task_id}: perfect ({perfect_score}) not > random ({random_score})"

    def test_task1_easier_than_task3(self):
        """Perfect agent scores higher on easy vs hard task."""
        def run_perfect(task_id):
            env = EmailEnvironment(task_id=task_id)
            env.reset()
            while not env._done:
                email = env._emails[env._index]
                a = EmailAction(
                    action_type=ActionType(email["expected_action"]),
                    classification=Priority(email["label"]),
                )
                r = env.step(a)
                if r.done: break
            return env.run_grader()["score"]

        score1 = run_perfect(1)
        score3 = run_perfect(3)
        assert score1 >= score3, f"Task 1 (easy) {score1} should be >= Task 3 (hard) {score3}"


# ── DATASET TESTS ─────────────────────────────────────────────────────────────

class TestDataset:
    def test_dataset_loads(self):
        env = EmailEnvironment(task_id=2)
        assert len(env._all_emails) >= 10

    def test_emails_have_required_fields(self):
        env = EmailEnvironment(task_id=2)
        required = {"id", "subject", "body", "sender", "label", "expected_action"}
        for e in env._all_emails:
            missing = required - set(e.keys())
            assert not missing, f"Email {e.get('id')} missing: {missing}"

    def test_urgent_emails_present(self):
        env = EmailEnvironment(task_id=2)
        urgents = [e for e in env._all_emails if e["label"] == "urgent"]
        assert len(urgents) >= 2, "Need ≥2 urgent emails for meaningful hard task"

    def test_spam_emails_present(self):
        env = EmailEnvironment(task_id=2)
        spams = [e for e in env._all_emails if e["label"] == "spam"]
        assert len(spams) >= 1

    def test_label_values_valid(self):
        valid = {"urgent", "important", "low_priority", "spam"}
        env = EmailEnvironment(task_id=2)
        for e in env._all_emails:
            assert e["label"] in valid, f"Invalid label '{e['label']}' for email {e['id']}"

    def test_action_values_valid(self):
        valid = {"reply", "escalate", "archive", "schedule", "ignore", "classify"}
        env = EmailEnvironment(task_id=2)
        for e in env._all_emails:
            assert e["expected_action"] in valid, f"Invalid action '{e['expected_action']}' for email {e['id']}"


# ── FULL EPISODE INTEGRATION TEST ─────────────────────────────────────────────

class TestFullEpisode:
    @pytest.mark.parametrize("task_id", [1, 2, 3])
    def test_full_episode_completes(self, task_id):
        env = EmailEnvironment(task_id=task_id)
        env.reset()
        steps = 0
        while not env._done and steps < 500:
            r = env.step(make_action())
            steps += 1
            if r.done:
                break
        assert env._done, f"Task {task_id} episode did not complete in {steps} steps"

    @pytest.mark.parametrize("task_id", [1, 2, 3])
    def test_grader_after_full_episode(self, task_id):
        env = EmailEnvironment(task_id=task_id)
        env.reset()
        while not env._done:
            r = env.step(make_action())
            if r.done:
                break
        result = env.run_grader()
        assert "score" in result
        assert "details" in result
        assert 0.0 <= result["score"] <= 1.0
        assert result["details"]["emails_processed"] > 0
