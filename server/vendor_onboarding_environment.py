"""Core environment for vendor onboarding workflow review."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from ..grading import (
        TaskFixture,
        TaskGrade,
        get_panel,
        get_task,
        get_task_ids,
        get_task_ids_for_difficulty,
        grade_task,
        grader_progress,
        load_task_fixtures,
    )
    from ..models import (
        AVAILABLE_DECISIONS,
        AVAILABLE_FINDINGS,
        VendorOnboardingAction,
        VendorOnboardingObservation,
        VendorOnboardingRewardBreakdown,
        VendorOnboardingState,
    )
except ImportError:  # pragma: no cover
    from grading import (
        TaskFixture,
        TaskGrade,
        get_panel,
        get_task,
        get_task_ids,
        get_task_ids_for_difficulty,
        grade_task,
        grader_progress,
        load_task_fixtures,
    )
    from models import (
        AVAILABLE_DECISIONS,
        AVAILABLE_FINDINGS,
        VendorOnboardingAction,
        VendorOnboardingObservation,
        VendorOnboardingRewardBreakdown,
        VendorOnboardingState,
    )


class VendorOnboardingEnvironment(
    Environment[
        VendorOnboardingAction,
        VendorOnboardingObservation,
        VendorOnboardingState,
    ]
):
    """Deterministic vendor onboarding review environment."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._tasks = load_task_fixtures()
        self._task_ids = tuple(get_task_ids())
        self._task_cursor = 0
        self._current_task: TaskFixture | None = None
        self._state = VendorOnboardingState()

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> VendorOnboardingObservation:
        """Reset into a fresh deterministic vendor dossier."""
        task = self._select_task(seed=seed, **kwargs)
        self._current_task = task
        self._state = VendorOnboardingState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task.id,
            task_title=task.title,
            task_difficulty=task.difficulty,
            opened_targets=[],
            logged_findings=[],
            draft_decision=None,
            current_target_id=None,
            last_action_type=None,
            last_feedback="Review the dossier, gather evidence, and submit a decision.",
            remaining_steps=task.max_steps,
            submitted=False,
            timeout_reached=False,
            final_score=None,
        )
        reward_breakdown = VendorOnboardingRewardBreakdown(
            notes=["Episode reset to a clean vendor review state."]
        )
        return self._build_observation(
            reward=0.0,
            done=False,
            reward_breakdown=reward_breakdown,
        )

    def step(
        self,
        action: VendorOnboardingAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> VendorOnboardingObservation:
        """Execute one structured workflow action."""
        del timeout_s, kwargs

        if self._current_task is None:
            # HTTP simulation mode creates a fresh environment instance per request,
            # so allow step() to bootstrap a clean default episode on first use.
            self.reset()

        self._state.step_count += 1
        self._state.last_action_type = action.action_type
        self._state.remaining_steps = max(
            self._current_task.max_steps - self._state.step_count,
            0,
        )

        reward_breakdown = VendorOnboardingRewardBreakdown()
        grade: TaskGrade | None = None
        current_target_content: dict[str, Any] | None = None
        done = False

        if action.action_type == "inspect":
            feedback, current_target_content = self._handle_inspect(action, reward_breakdown)
        elif action.action_type == "record_finding":
            feedback = self._handle_record_finding(action, reward_breakdown)
        elif action.action_type == "set_decision":
            feedback = self._handle_set_decision(action, reward_breakdown)
        else:
            feedback, grade = self._handle_submit(reward_breakdown)
            done = True

        if not done and self._state.remaining_steps == 0:
            feedback, grade = self._handle_timeout(reward_breakdown)
            done = True

        reward_breakdown.total_reward = round(
            reward_breakdown.inspection_reward
            + reward_breakdown.finding_reward
            + reward_breakdown.decision_reward
            + reward_breakdown.submission_reward
            + reward_breakdown.penalty
            + reward_breakdown.timeout_penalty,
            4,
        )

        self._state.last_feedback = feedback
        return self._build_observation(
            reward=reward_breakdown.total_reward,
            done=done,
            reward_breakdown=reward_breakdown,
            grade=grade,
            current_target_content=current_target_content,
        )

    @property
    def state(self) -> VendorOnboardingState:
        """Return current non-secret environment state."""
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        """Provide environment metadata for OpenEnv discovery."""
        return EnvironmentMetadata(
            name="vendor_onboarding_env",
            description=(
                "Structured vendor onboarding and compliance review with "
                "deterministic dossiers, shaped rewards, and programmatic grading."
            ),
            version="0.1.0",
        )

    def _select_task(
        self,
        seed: int | None = None,
        task_id: str | None = None,
        difficulty: str | None = None,
    ) -> TaskFixture:
        """Select a deterministic task fixture."""
        if task_id:
            return get_task(task_id)

        if difficulty:
            candidates = get_task_ids_for_difficulty(difficulty)
            if not candidates:
                raise ValueError(f"No tasks found for difficulty '{difficulty}'.")
            return get_task(candidates[0])

        if seed is not None:
            return get_task(self._task_ids[seed % len(self._task_ids)])

        task = get_task(self._task_ids[self._task_cursor % len(self._task_ids)])
        self._task_cursor += 1
        return task

    def _handle_inspect(
        self,
        action: VendorOnboardingAction,
        reward_breakdown: VendorOnboardingRewardBreakdown,
    ) -> tuple[str, dict[str, Any] | None]:
        """Inspect a dossier panel and expose its content."""
        assert self._current_task is not None
        target_id = action.target_id or ""
        panel = get_panel(self._current_task, target_id)

        if panel is None:
            reward_breakdown.penalty -= 0.1
            reward_breakdown.notes.append("Tried to inspect an unknown dossier panel.")
            return (
                f"'{target_id}' is not a valid panel. Choose one of the available targets.",
                None,
            )

        if target_id in self._state.opened_targets:
            reward_breakdown.penalty -= 0.04
            reward_breakdown.notes.append(f"Re-inspected already opened panel '{target_id}'.")
        else:
            self._state.opened_targets.append(target_id)
            panel_reward = 0.07 if target_id in self._current_task.grader.critical_targets else 0.03
            reward_breakdown.inspection_reward += panel_reward
            reward_breakdown.notes.append(f"Opened dossier panel '{target_id}'.")

        self._state.current_target_id = target_id
        return (
            f"Opened {panel.title}. Review the details for discrepancies or policy blockers.",
            self._panel_to_content(panel),
        )

    def _handle_record_finding(
        self,
        action: VendorOnboardingAction,
        reward_breakdown: VendorOnboardingRewardBreakdown,
    ) -> str:
        """Record a normalized finding against the case."""
        assert self._current_task is not None
        finding_code = (action.finding_code or "").strip()

        if finding_code not in AVAILABLE_FINDINGS:
            reward_breakdown.penalty -= 0.1
            reward_breakdown.notes.append("Attempted to log a non-normalized finding code.")
            return (
                f"'{finding_code}' is not a supported finding code. "
                "Use one of the advertised normalized findings."
            )

        if finding_code in self._state.logged_findings:
            reward_breakdown.penalty -= 0.08
            reward_breakdown.notes.append(f"Duplicate finding '{finding_code}' was re-submitted.")
            return f"Finding '{finding_code}' was already recorded for this case."

        self._state.logged_findings.append(finding_code)
        if finding_code in self._current_task.grader.required_findings:
            reward_breakdown.finding_reward += 0.18
            reward_breakdown.notes.append(f"Correct finding '{finding_code}' recorded.")
            return f"Recorded '{finding_code}' as a material finding."

        if finding_code in self._current_task.grader.severe_false_findings:
            reward_breakdown.penalty -= 0.18
            reward_breakdown.notes.append(f"Severe false-positive finding '{finding_code}' recorded.")
            return f"Recorded '{finding_code}', but it does not match the evidence in this case."

        reward_breakdown.penalty -= 0.06
        reward_breakdown.notes.append(f"Irrelevant finding '{finding_code}' recorded.")
        return f"Recorded '{finding_code}', but it is not material for this task."

    def _handle_set_decision(
        self,
        action: VendorOnboardingAction,
        reward_breakdown: VendorOnboardingRewardBreakdown,
    ) -> str:
        """Set or revise the draft decision."""
        assert self._current_task is not None
        decision = action.decision
        assert decision is not None

        if self._state.draft_decision == decision:
            reward_breakdown.penalty -= 0.03
            reward_breakdown.notes.append("Repeated the same draft decision.")
            return f"Draft decision is already '{decision}'."

        self._state.draft_decision = decision
        correct = self._current_task.grader.correct_decision
        decision_rank = {"approve": 0, "escalate": 1, "reject": 2}

        if decision == correct:
            reward_breakdown.decision_reward += 0.1
            reward_breakdown.notes.append(f"Set the correct draft decision '{decision}'.")
            return f"Draft decision updated to '{decision}'."

        if decision_rank[decision] < decision_rank[correct]:
            penalty = 0.16 if decision == "approve" else 0.08
            reward_breakdown.penalty -= penalty
            reward_breakdown.notes.append(f"Unsafe under-escalation: chose '{decision}' instead of '{correct}'.")
            return (
                f"Draft decision '{decision}' looks too permissive for the evidence gathered so far."
            )

        reward_breakdown.penalty -= 0.04
        reward_breakdown.notes.append(f"Overly conservative decision '{decision}' set.")
        return f"Draft decision updated to '{decision}', but it appears more conservative than necessary."

    def _handle_submit(
        self,
        reward_breakdown: VendorOnboardingRewardBreakdown,
    ) -> tuple[str, TaskGrade]:
        """Finalize the case and compute the deterministic task score."""
        assert self._current_task is not None
        grade = grade_task(
            self._current_task,
            self._state.opened_targets,
            self._state.logged_findings,
            self._state.draft_decision,
        )
        self._state.submitted = True
        self._state.final_score = grade.final_score
        reward_breakdown.final_task_score = grade.final_score
        reward_breakdown.submission_reward += round(0.6 * grade.final_score, 4)
        reward_breakdown.notes.append("Episode submitted and graded.")

        if self._state.draft_decision is None:
            reward_breakdown.penalty -= 0.15
            reward_breakdown.notes.append("Submission occurred without any draft decision.")

        feedback = (
            f"Submission recorded. Final task score: {grade.final_score:.2f}. "
            f"Expected decision: {grade.correct_decision}."
        )
        return feedback, grade

    def _handle_timeout(
        self,
        reward_breakdown: VendorOnboardingRewardBreakdown,
    ) -> tuple[str, TaskGrade]:
        """Terminate the episode when the step limit is exhausted."""
        assert self._current_task is not None
        grade = grade_task(
            self._current_task,
            self._state.opened_targets,
            self._state.logged_findings,
            self._state.draft_decision,
        )
        self._state.timeout_reached = True
        self._state.final_score = grade.final_score
        reward_breakdown.final_task_score = grade.final_score
        reward_breakdown.submission_reward += round(0.25 * grade.final_score, 4)
        reward_breakdown.timeout_penalty -= 0.25
        reward_breakdown.notes.append("Episode terminated after reaching the max step budget.")
        feedback = (
            f"Maximum step budget reached. Episode closed with task score {grade.final_score:.2f}."
        )
        return feedback, grade

    def _build_observation(
        self,
        reward: float,
        done: bool,
        reward_breakdown: VendorOnboardingRewardBreakdown,
        grade: TaskGrade | None = None,
        current_target_content: dict[str, Any] | None = None,
    ) -> VendorOnboardingObservation:
        """Create the agent-facing observation."""
        assert self._current_task is not None
        if current_target_content is None and self._state.current_target_id:
            panel = get_panel(self._current_task, self._state.current_target_id)
            if panel is not None:
                current_target_content = self._panel_to_content(panel)

        metadata: dict[str, Any] = {
            "info": {
                "reward_breakdown": reward_breakdown.model_dump(),
                "grader_progress": grader_progress(
                    self._current_task,
                    self._state.opened_targets,
                    self._state.logged_findings,
                    self._state.draft_decision,
                ),
            }
        }
        if grade is not None:
            metadata["info"]["task_grade"] = grade.model_dump()

        return VendorOnboardingObservation(
            task_id=self._current_task.id,
            task_title=self._current_task.title,
            task_difficulty=self._current_task.difficulty,
            objective=self._current_task.objective,
            available_targets=[panel.id for panel in self._current_task.panels],
            available_findings=list(AVAILABLE_FINDINGS),
            available_decisions=list(AVAILABLE_DECISIONS),
            opened_targets=list(self._state.opened_targets),
            current_target_id=self._state.current_target_id,
            current_target_content=current_target_content,
            logged_findings=list(self._state.logged_findings),
            draft_decision=self._state.draft_decision,
            step_count=self._state.step_count,
            remaining_steps=self._state.remaining_steps,
            last_feedback=self._state.last_feedback,
            reward=reward,
            done=done,
            metadata=metadata,
        )

    @staticmethod
    def _panel_to_content(panel: Any) -> dict[str, Any]:
        """Normalize panel payloads for observations."""
        return {
            "id": panel.id,
            "title": panel.title,
            "summary": panel.summary,
            "details": list(panel.details),
        }
