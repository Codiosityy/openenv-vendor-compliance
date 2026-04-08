---
title: Vendor Onboarding Env
emoji: 📋
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
---

# Vendor Onboarding Env

`vendor_onboarding_env` is a real-world OpenEnv environment for vendor onboarding and compliance review. The agent behaves like an operations analyst reviewing a vendor dossier, opening supporting panels, logging normalized findings, choosing a decision, and submitting the case for deterministic scoring.

This environment is designed for agent evaluation rather than toy play. It models a workflow humans actually do: vendor activation review across legal identity, tax documentation, bank ownership, beneficial ownership, sanctions screening, prior analyst notes, and policy guidance.

## Motivation

Vendor onboarding is practical, safety-sensitive, and naturally multi-step. An agent has to gather evidence before acting, avoid false positives, and make an approval, escalation, or rejection decision that matches documented policy.

The environment is benchmark-friendly:

- 3 fixed tasks with easy, medium, and hard difficulty
- programmatic graders that always return a score from `0.0` to `1.0`
- shaped rewards during the trajectory, not just at episode end
- clean reset semantics with reproducible task fixtures in JSON

## Action Space

`VendorOnboardingAction` supports four workflow actions:

- `inspect`: open a dossier panel by `target_id`
- `record_finding`: log a normalized `finding_code`
- `set_decision`: set a draft decision to `approve`, `escalate`, or `reject`
- `submit`: finalize the case and trigger deterministic grading

Accepted finding codes:

- `missing_required_doc`
- `bank_owner_mismatch`
- `sanctions_exact_match`
- `ubo_inconsistency`
- `address_mismatch`
- `name_mismatch`

## Observation Space

`VendorOnboardingObservation` exposes agent-safe state only:

- task metadata: id, title, difficulty, objective
- available targets, findings, and decisions
- already opened panels
- current panel content
- logged findings
- draft decision
- step count and remaining steps
- last action feedback

`VendorOnboardingState` exposes the current non-secret episode state returned by `state()`.

Reward details and grader progress are returned in `observation.metadata["info"]` because the current OpenEnv HTTP response shape provides `observation`, `reward`, and `done`, while extra structured diagnostics live in observation metadata.

## Task Set

### Easy: `easy-clean-approve`

A clean SaaS vendor dossier. The agent should inspect the core verification panels and approve the vendor without fabricating severe findings.

### Medium: `medium-bank-mismatch`

The vendor's bank confirmation letter names a different account owner than the legal entity. The agent must identify `bank_owner_mismatch` and escalate the case.

### Hard: `hard-sanctioned-ubo-reject`

The controlling beneficial owner has a corroborated sanctions exact match. The agent must inspect ownership, sanctions, analyst notes, and policy, then record both `sanctions_exact_match` and `ubo_inconsistency` before rejecting the case.

## Reward Design

Per-step reward is shaped to provide signal across the full trajectory:

- first-time inspection of task-critical evidence: positive reward
- correct first-time finding: positive reward
- correct draft decision: positive reward
- duplicate actions, unsupported findings, or unsafe under-escalation: penalties
- submit: terminal reward proportional to the final deterministic task score
- timeout: episode closes with a penalty and a partial task score snapshot

Final task score is deterministic and separate from the shaped trajectory reward:

- decision correctness: `0.5`
- required findings coverage: `0.3`
- required evidence coverage: `0.2`
- penalties for severe false findings and irrelevant findings

## Local Setup

Install the project in editable mode:

```bash
python -m pip install -e .[dev]
```

Run the server locally:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Validate the environment:

```bash
openenv validate
```

Run the tests:

```bash
pytest
```

## Docker

Build from the repo root:

```bash
docker build -t vendor-onboarding-env:latest .
```


Run:

```bash
docker run -p 7860:7860 vendor-onboarding-env:latest
```

The container listens on port `7860` by default (for HF Spaces compatibility). Override with the `PORT` environment variable:

```bash
docker run -p 8000:8000 -e PORT=8000 vendor-onboarding-env:latest
```

## Hugging Face Spaces

This repository is structured as a single OpenEnv environment at the repo root and is intended to be pushed as a Docker-based FastAPI Space:

```bash
openenv push
```

The manifest includes the `openenv` tag and a workflow/compliance-focused description.

## Baseline Inference

The mandatory inference script is at the project root (`inference.py`). It follows the OpenEnv submission format with structured `[START]`/`[STEP]`/`[END]` stdout logging.

Required environment variables:
- `HF_TOKEN` or `API_KEY` — Your API key (no default)
- `API_BASE_URL` — LLM endpoint (default: `https://router.huggingface.co/v1`)
- `MODEL_NAME` — Model identifier (default: `Qwen/Qwen2.5-72B-Instruct`)
- `IMAGE_NAME` — Docker image name (optional, for `from_docker_image()`)

Run against a local server:

```bash
HF_TOKEN=... python inference.py
```

Or use the extended baseline runner with result persistence:

```bash
HF_TOKEN=... python examples/run_baseline.py --base-url http://localhost:8000
```

## Baseline Scores

Estimated baseline scores using a heuristic fallback agent (inspect all panels → escalate → submit):

| Task | Difficulty | Score | Notes |
|------|-----------|-------|-------|
| `easy-clean-approve` | Easy | 0.50 | Heuristic escalates instead of approving; loses decision weight |
| `medium-bank-mismatch` | Medium | 0.70 | Correct escalation but missing `bank_owner_mismatch` finding |
| `hard-sanctioned-ubo-reject` | Hard | 0.20 | Heuristic escalates instead of rejecting; misses both findings |
| **Aggregate** | — | **0.47** | Average across all 3 tasks |

With a frontier LLM (e.g., Qwen2.5-72B-Instruct or GPT-4.1), scores are expected to improve significantly as the model can read panel content and make informed decisions. Run `inference.py` with a valid `HF_TOKEN` to produce real baseline scores.
