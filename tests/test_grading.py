"""Deterministic grader tests."""

from vendor_onboarding_env.grading import get_task, grade_task


def test_medium_task_correct_path_scores_full_credit() -> None:
    task = get_task("medium-bank-mismatch")
    grade = grade_task(
        task,
        opened_targets=["overview", "bank_letter"],
        logged_findings=["bank_owner_mismatch"],
        draft_decision="escalate",
    )
    assert grade.final_score == 1.0


def test_medium_task_partial_credit_is_deterministic() -> None:
    task = get_task("medium-bank-mismatch")
    grade = grade_task(
        task,
        opened_targets=["overview"],
        logged_findings=[],
        draft_decision="escalate",
    )
    assert grade.final_score == 0.6


def test_unsafe_medium_submission_scores_low() -> None:
    task = get_task("medium-bank-mismatch")
    grade = grade_task(
        task,
        opened_targets=["overview", "bank_letter"],
        logged_findings=["name_mismatch"],
        draft_decision="approve",
    )
    assert grade.final_score == 0.15

