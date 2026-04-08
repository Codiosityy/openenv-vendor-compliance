"""Typed models for the vendor onboarding environment."""

from __future__ import annotations

from typing import Any, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, ConfigDict, Field, model_validator

ActionType = Literal["inspect", "record_finding", "set_decision", "submit"]
DecisionType = Literal["approve", "escalate", "reject"]
DifficultyType = Literal["easy", "medium", "hard"]

AVAILABLE_FINDINGS = [
    "missing_required_doc",
    "bank_owner_mismatch",
    "sanctions_exact_match",
    "ubo_inconsistency",
    "address_mismatch",
    "name_mismatch",
]

AVAILABLE_DECISIONS = ["approve", "escalate", "reject"]


class VendorOnboardingAction(Action):
    """Structured actions available to the agent."""

    action_type: ActionType = Field(
        ..., description="Type of action to execute in the dossier workflow."
    )
    target_id: str | None = Field(
        default=None,
        description="Panel identifier to inspect for inspect actions.",
    )
    finding_code: str | None = Field(
        default=None,
        description="Normalized finding code for record_finding actions.",
    )
    decision: DecisionType | None = Field(
        default=None,
        description="Draft decision for set_decision actions.",
    )

    @model_validator(mode="after")
    def validate_action_fields(self) -> "VendorOnboardingAction":
        """Require only the fields relevant to the selected action."""
        required_by_action = {
            "inspect": {"target_id"},
            "record_finding": {"finding_code"},
            "set_decision": {"decision"},
            "submit": set(),
        }
        provided = {
            field_name
            for field_name in ("target_id", "finding_code", "decision")
            if getattr(self, field_name) is not None
        }
        required = required_by_action[self.action_type]

        missing = required - provided
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(
                f"{self.action_type} actions require the following field(s): {missing_list}"
            )

        unexpected = provided - required
        if unexpected:
            unexpected_list = ", ".join(sorted(unexpected))
            raise ValueError(
                f"{self.action_type} actions do not accept: {unexpected_list}"
            )

        return self


class VendorOnboardingRewardBreakdown(BaseModel):
    """Typed reward details surfaced through observation metadata."""

    model_config = ConfigDict(extra="forbid")

    inspection_reward: float = Field(default=0.0)
    finding_reward: float = Field(default=0.0)
    decision_reward: float = Field(default=0.0)
    submission_reward: float = Field(default=0.0)
    penalty: float = Field(default=0.0)
    timeout_penalty: float = Field(default=0.0)
    final_task_score: float | None = Field(
        default=None,
        description="Deterministic 0.0-1.0 task score when the episode terminates.",
    )
    total_reward: float = Field(default=0.0)
    notes: list[str] = Field(default_factory=list)


class VendorOnboardingObservation(Observation):
    """Agent-safe observation of the current dossier review state."""

    task_id: str = Field(..., description="Unique task identifier.")
    task_title: str = Field(..., description="Human-readable task title.")
    task_difficulty: DifficultyType = Field(..., description="Task difficulty tier.")
    objective: str = Field(..., description="Concrete agent objective for the episode.")
    available_targets: list[str] = Field(
        default_factory=list,
        description="Inspectable dossier panels.",
    )
    available_findings: list[str] = Field(
        default_factory=lambda: list(AVAILABLE_FINDINGS),
        description="Normalized finding codes accepted by the environment.",
    )
    available_decisions: list[str] = Field(
        default_factory=lambda: list(AVAILABLE_DECISIONS),
        description="Allowed final decisions.",
    )
    opened_targets: list[str] = Field(
        default_factory=list,
        description="Panels the agent has already inspected.",
    )
    current_target_id: str | None = Field(
        default=None,
        description="Currently selected panel identifier, if any.",
    )
    current_target_content: dict[str, Any] | None = Field(
        default=None,
        description="Structured content for the currently selected panel.",
    )
    logged_findings: list[str] = Field(
        default_factory=list,
        description="Findings the agent has already recorded.",
    )
    draft_decision: DecisionType | None = Field(
        default=None,
        description="Current draft decision.",
    )
    step_count: int = Field(..., ge=0, description="Steps taken this episode.")
    remaining_steps: int = Field(
        ..., ge=0, description="Remaining step budget before timeout."
    )
    last_feedback: str = Field(
        default="",
        description="Short feedback message about the latest action outcome.",
    )


class VendorOnboardingState(State):
    """Current non-secret environment state."""

    task_id: str | None = Field(default=None)
    task_title: str | None = Field(default=None)
    task_difficulty: DifficultyType | None = Field(default=None)
    opened_targets: list[str] = Field(default_factory=list)
    logged_findings: list[str] = Field(default_factory=list)
    draft_decision: DecisionType | None = Field(default=None)
    current_target_id: str | None = Field(default=None)
    last_action_type: ActionType | None = Field(default=None)
    last_feedback: str = Field(default="")
    remaining_steps: int = Field(default=0, ge=0)
    submitted: bool = Field(default=False)
    timeout_reached: bool = Field(default=False)
    final_score: float | None = Field(default=None)

