"""
Inference Script — Vendor Onboarding Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    IMAGE_NAME     The name of the local image to use for the environment if you are using from_docker_image()

- Defaults are set only for API_BASE_URL and MODEL_NAME:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]
"""

import asyncio
import json
import os
import textwrap
from typing import Any, List, Optional

from openai import OpenAI

from vendor_onboarding_env import VendorOnboardingAction, VendorOnboardingEnv
from vendor_onboarding_env.grading import get_task_ids

# ---------------------------------------------------------------------------
# Environment variables — MANDATORY
# ---------------------------------------------------------------------------
IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK = "vendor_onboarding_env"
MAX_STEPS = 12  # matches task max_steps
TEMPERATURE = 0.0
MAX_TOKENS = 300

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an operations analyst reviewing vendor onboarding dossiers.
    Return exactly one JSON object matching this schema:
    {
      "action_type": "inspect" | "record_finding" | "set_decision" | "submit",
      "target_id": string | null,
      "finding_code": string | null,
      "decision": "approve" | "escalate" | "reject" | null
    }

    Rules:
    - Inspect panels before deciding.
    - Use only finding codes and decisions listed in the observation.
    - Use null for fields that are not relevant to the chosen action.
    - Never include markdown or explanations outside the JSON object.
    """
).strip()


# ---------------------------------------------------------------------------
# Structured stdout helpers
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def _build_user_prompt(observation: dict[str, Any]) -> str:
    return json.dumps(
        {
            "task_id": observation.get("task_id"),
            "task_title": observation.get("task_title"),
            "difficulty": observation.get("task_difficulty"),
            "objective": observation.get("objective"),
            "available_targets": observation.get("available_targets"),
            "available_findings": observation.get("available_findings"),
            "available_decisions": observation.get("available_decisions"),
            "opened_targets": observation.get("opened_targets"),
            "logged_findings": observation.get("logged_findings"),
            "draft_decision": observation.get("draft_decision"),
            "step_count": observation.get("step_count"),
            "remaining_steps": observation.get("remaining_steps"),
            "last_feedback": observation.get("last_feedback"),
            "current_target_content": observation.get("current_target_content"),
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# Fallback heuristic (used when LLM call fails)
# ---------------------------------------------------------------------------
def _fallback_action(observation: dict[str, Any]) -> VendorOnboardingAction:
    unopened = [
        t
        for t in observation.get("available_targets", [])
        if t not in observation.get("opened_targets", [])
    ]
    if unopened:
        return VendorOnboardingAction(action_type="inspect", target_id=unopened[0])
    if observation.get("draft_decision") is None:
        return VendorOnboardingAction(action_type="set_decision", decision="escalate")
    return VendorOnboardingAction(action_type="submit")


# ---------------------------------------------------------------------------
# LLM action request
# ---------------------------------------------------------------------------
def _request_action(
    client: OpenAI,
    model: str,
    observation: dict[str, Any],
) -> VendorOnboardingAction:
    user_prompt = _build_user_prompt(observation)
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw_text = response.choices[0].message.content or "{}"
        parsed = json.loads(raw_text)
        return VendorOnboardingAction.model_validate(parsed)
    except Exception as exc:
        print(f"[DEBUG] Model action parse failed, falling back to heuristic: {exc}", flush=True)
        return _fallback_action(observation)


# ---------------------------------------------------------------------------
# Action to short string for logging
# ---------------------------------------------------------------------------
def _action_str(action: VendorOnboardingAction) -> str:
    parts = [action.action_type]
    if action.target_id:
        parts.append(f"target_id={action.target_id}")
    if action.finding_code:
        parts.append(f"finding_code={action.finding_code}")
    if action.decision:
        parts.append(f"decision={action.decision}")
    return ",".join(parts)


# ---------------------------------------------------------------------------
# Connect to environment
# ---------------------------------------------------------------------------
async def _connect_env() -> VendorOnboardingEnv:
    if IMAGE_NAME:
        return await VendorOnboardingEnv.from_docker_image(IMAGE_NAME)

    base_url = os.getenv("VENDOR_ONBOARDING_BASE_URL", "http://localhost:8000")
    env = VendorOnboardingEnv(base_url=base_url)
    await env.connect()
    return env


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await _connect_env()

    task_ids = get_task_ids()

    try:
        for task_id in task_ids:
            rewards: List[float] = []
            steps_taken = 0
            score = 0.0
            success = False

            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

            try:
                result = await env.reset(task_id=task_id)

                for step in range(1, MAX_STEPS + 1):
                    if result.done:
                        break

                    obs_dict = result.observation.model_dump()
                    action = _request_action(client, MODEL_NAME, obs_dict)
                    result = await env.step(action)

                    reward = result.reward or 0.0
                    done = result.done
                    error = None

                    rewards.append(float(reward))
                    steps_taken = step

                    log_step(
                        step=step,
                        action=_action_str(action),
                        reward=float(reward),
                        done=done,
                        error=error,
                    )

                    if done:
                        info = result.observation.metadata.get("info", {})
                        score = float(
                            info.get("task_grade", {}).get("final_score", 0.0)
                        )
                        break

                # Clamp score to [0, 1]
                score = min(max(score, 0.0), 1.0)
                success = score > 0.0

            finally:
                log_end(
                    success=success,
                    steps=steps_taken,
                    score=score,
                    rewards=rewards,
                )

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
