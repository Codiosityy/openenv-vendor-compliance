"""OpenAI baseline runner for the vendor onboarding environment."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from vendor_onboarding_env import VendorOnboardingAction, VendorOnboardingEnv
from vendor_onboarding_env.grading import get_task_ids

DEFAULT_IMAGE = "vendor-onboarding-env:latest"
DEFAULT_BASE_URL = "http://localhost:8000"
SYSTEM_PROMPT = """You are an operations analyst reviewing vendor onboarding dossiers.
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


def _build_user_prompt(observation: dict[str, Any]) -> str:
    return json.dumps(
        {
            "task_id": observation["task_id"],
            "task_title": observation["task_title"],
            "difficulty": observation["task_difficulty"],
            "objective": observation["objective"],
            "available_targets": observation["available_targets"],
            "available_findings": observation["available_findings"],
            "available_decisions": observation["available_decisions"],
            "opened_targets": observation["opened_targets"],
            "logged_findings": observation["logged_findings"],
            "draft_decision": observation["draft_decision"],
            "step_count": observation["step_count"],
            "remaining_steps": observation["remaining_steps"],
            "last_feedback": observation["last_feedback"],
            "current_target_content": observation["current_target_content"],
        },
        indent=2,
    )


def _fallback_action(observation: dict[str, Any]) -> VendorOnboardingAction:
    unopened = [
        target
        for target in observation["available_targets"]
        if target not in observation["opened_targets"]
    ]
    if unopened:
        return VendorOnboardingAction(action_type="inspect", target_id=unopened[0])
    if observation["draft_decision"] is None:
        return VendorOnboardingAction(action_type="set_decision", decision="escalate")
    return VendorOnboardingAction(action_type="submit")


def _request_action(
    client: OpenAI,
    model: str,
    observation: dict[str, Any],
) -> VendorOnboardingAction:
    user_prompt = _build_user_prompt(observation)
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
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
        print(f"[WARN] Model action parse failed, falling back to heuristic: {exc}")
        return _fallback_action(observation)


async def _connect_env(
    base_url: str | None,
    docker_image: str | None,
) -> VendorOnboardingEnv:
    if docker_image:
        return await VendorOnboardingEnv.from_docker_image(docker_image)

    env = VendorOnboardingEnv(base_url=base_url or DEFAULT_BASE_URL)
    await env.connect()
    return env


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("VENDOR_ONBOARDING_BASE_URL"))
    parser.add_argument("--docker-image", default=os.environ.get("VENDOR_ONBOARDING_IMAGE"))
    parser.add_argument("--output", default="outputs/evals/baseline_results.json")
    args = parser.parse_args()

    api_key = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY")
    if not api_key:
        raise RuntimeError(
            "HF_TOKEN or API_KEY is required to run the baseline. "
            "Set one of these environment variables."
        )

    model = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    api_base_url = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
    client = OpenAI(api_key=api_key, base_url=api_base_url)

    docker_image = args.docker_image or (None if args.base_url else DEFAULT_IMAGE)
    env = await _connect_env(args.base_url, docker_image)
    task_results: list[dict[str, Any]] = []

    try:
        for task_id in get_task_ids():
            print(f"\n=== Running task: {task_id} ===")
            result = await env.reset(task_id=task_id)
            rewards: list[float] = []
            final_score = 0.0
            actions: list[dict[str, Any]] = []

            while not result.done:
                action = _request_action(client, model, result.observation.model_dump())
                actions.append(action.model_dump(exclude_none=True))
                result = await env.step(action)
                rewards.append(float(result.reward or 0.0))
                print(
                    f"step={result.observation.step_count} "
                    f"action={action.model_dump(exclude_none=True)} "
                    f"reward={result.reward} done={result.done}"
                )
                if result.done:
                    info = result.observation.metadata.get("info", {})
                    final_score = float(info.get("task_grade", {}).get("final_score", 0.0))

            task_results.append(
                {
                    "task_id": task_id,
                    "final_score": final_score,
                    "total_reward": round(sum(rewards), 4),
                    "steps": len(rewards),
                    "actions": actions,
                }
            )

        aggregate_score = round(
            sum(result["final_score"] for result in task_results) / len(task_results),
            4,
        )
        summary = {"model": model, "aggregate_score": aggregate_score, "tasks": task_results}
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2))
        print(f"\nAggregate normalized score: {aggregate_score:.4f}")
        print(f"Saved baseline results to {output_path}")
    finally:
        await env.close()


if __name__ == "__main__":
    asyncio.run(main())
