"""Task fixtures and deterministic grading for vendor onboarding."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:
    from .models import DifficultyType
except ImportError:  # pragma: no cover - supports direct source imports
    from models import DifficultyType


class PanelSpec(BaseModel):
    """Inspectable panel in a vendor dossier."""

    id: str
    title: str
    summary: str
    details: list[str] = Field(default_factory=list)


class GradingSpec(BaseModel):
    """Hidden deterministic grading configuration."""

    correct_decision: str
    required_findings: list[str] = Field(default_factory=list)
    severe_false_findings: list[str] = Field(default_factory=list)
    required_targets: list[str] = Field(default_factory=list)
    critical_targets: list[str] = Field(default_factory=list)
    decision_weight: float = 0.5
    findings_weight: float = 0.3
    evidence_weight: float = 0.2


class TaskFixture(BaseModel):
    """Complete vendor review task fixture."""

    id: str
    title: str
    difficulty: DifficultyType
    objective: str
    max_steps: int = 12
    panels: list[PanelSpec]
    grader: GradingSpec


class FixtureBundle(BaseModel):
    """Top-level JSON fixture payload."""

    tasks: list[TaskFixture]


class TaskGrade(BaseModel):
    """Deterministic task score details."""

    model_config = ConfigDict(extra="forbid")

    final_score: float
    decision_score: float
    findings_score: float
    evidence_score: float
    required_findings_found: int
    required_findings_total: int
    required_targets_opened: int
    required_targets_total: int
    missing_required_findings: list[str] = Field(default_factory=list)
    missing_required_targets: list[str] = Field(default_factory=list)
    severe_false_findings: list[str] = Field(default_factory=list)
    irrelevant_findings: list[str] = Field(default_factory=list)
    correct_decision: str


def _fixture_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "tasks.json"


@lru_cache(maxsize=1)
def load_task_fixtures() -> dict[str, TaskFixture]:
    """Load and cache deterministic task fixtures."""
    bundle = FixtureBundle.model_validate(json.loads(_fixture_path().read_text()))
    return {task.id: task for task in bundle.tasks}


def get_task(task_id: str) -> TaskFixture:
    """Return a specific task fixture by id."""
    tasks = load_task_fixtures()
    if task_id not in tasks:
        raise KeyError(f"Unknown task id: {task_id}")
    return tasks[task_id]


def get_task_ids() -> list[str]:
    """Return task ids in fixture order."""
    return list(load_task_fixtures().keys())


def get_task_ids_for_difficulty(difficulty: str) -> list[str]:
    """Return task ids for a specific difficulty tier."""
    return [
        task.id
        for task in load_task_fixtures().values()
        if task.difficulty == difficulty
    ]


def get_panel(task: TaskFixture, target_id: str) -> PanelSpec | None:
    """Fetch a panel by id from a task."""
    for panel in task.panels:
        if panel.id == target_id:
            return panel
    return None


def grade_task(
    task: TaskFixture,
    opened_targets: list[str],
    logged_findings: list[str],
    draft_decision: str | None,
) -> TaskGrade:
    """Compute the final deterministic 0.0-1.0 task score from final state."""
    opened = set(opened_targets)
    findings = set(logged_findings)
    required_findings = set(task.grader.required_findings)
    required_targets = set(task.grader.required_targets)
    severe_false = set(task.grader.severe_false_findings)

    found_required = findings & required_findings
    missing_required_findings = sorted(required_findings - findings)
    opened_required = opened & required_targets
    missing_required_targets = sorted(required_targets - opened)
    severe_false_hits = sorted(findings & severe_false)
    irrelevant_findings = sorted(
        finding
        for finding in findings
        if finding not in required_findings and finding not in severe_false
    )

    decision_score = 1.0 if draft_decision == task.grader.correct_decision else 0.0
    findings_score = (
        len(found_required) / len(required_findings) if required_findings else 1.0
    )
    evidence_score = (
        len(opened_required) / len(required_targets) if required_targets else 1.0
    )

    score = (
        task.grader.decision_weight * decision_score
        + task.grader.findings_weight * findings_score
        + task.grader.evidence_weight * evidence_score
    )
    score -= 0.15 * len(severe_false_hits)
    score -= 0.05 * len(irrelevant_findings)
    score = max(0.0, min(1.0, round(score, 4)))

    return TaskGrade(
        final_score=score,
        decision_score=round(decision_score, 4),
        findings_score=round(findings_score, 4),
        evidence_score=round(evidence_score, 4),
        required_findings_found=len(found_required),
        required_findings_total=len(required_findings),
        required_targets_opened=len(opened_required),
        required_targets_total=len(required_targets),
        missing_required_findings=missing_required_findings,
        missing_required_targets=missing_required_targets,
        severe_false_findings=severe_false_hits,
        irrelevant_findings=irrelevant_findings,
        correct_decision=task.grader.correct_decision,
    )


def grader_progress(
    task: TaskFixture,
    opened_targets: list[str],
    logged_findings: list[str],
    draft_decision: str | None,
) -> dict[str, Any]:
    """Return non-secret progress counters for logging and inspection."""
    grade = grade_task(task, opened_targets, logged_findings, draft_decision)
    return {
        "decision_set": draft_decision is not None,
        "required_findings_found": grade.required_findings_found,
        "required_findings_total": grade.required_findings_total,
        "required_targets_opened": grade.required_targets_opened,
        "required_targets_total": grade.required_targets_total,
        "logged_findings_count": len(logged_findings),
        "opened_targets_count": len(opened_targets),
        "irrelevant_findings_count": len(grade.irrelevant_findings),
        "severe_false_findings_count": len(grade.severe_false_findings),
    }
