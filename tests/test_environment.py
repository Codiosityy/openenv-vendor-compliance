"""Direct environment tests."""

from vendor_onboarding_env import VendorOnboardingAction
from vendor_onboarding_env.server.vendor_onboarding_environment import (
    VendorOnboardingEnvironment,
)


def test_easy_task_full_run_reaches_perfect_score() -> None:
    env = VendorOnboardingEnvironment()
    env.reset(task_id="easy-clean-approve")
    for target in [
        "overview",
        "tax_form",
        "bank_letter",
        "beneficial_owners",
        "sanctions_screen",
        "policy",
    ]:
        env.step(VendorOnboardingAction(action_type="inspect", target_id=target))
    env.step(VendorOnboardingAction(action_type="set_decision", decision="approve"))
    final_obs = env.step(VendorOnboardingAction(action_type="submit"))

    assert final_obs.done is True
    assert final_obs.metadata["info"]["task_grade"]["final_score"] == 1.0


def test_duplicate_inspect_is_penalized() -> None:
    env = VendorOnboardingEnvironment()
    env.reset(task_id="easy-clean-approve")
    env.step(VendorOnboardingAction(action_type="inspect", target_id="overview"))
    duplicate = env.step(VendorOnboardingAction(action_type="inspect", target_id="overview"))

    assert duplicate.reward < 0
    assert duplicate.metadata["info"]["reward_breakdown"]["penalty"] < 0


def test_reset_clears_previous_episode_state() -> None:
    env = VendorOnboardingEnvironment()
    env.reset(task_id="medium-bank-mismatch")
    env.step(VendorOnboardingAction(action_type="inspect", target_id="overview"))
    assert env.state.opened_targets == ["overview"]

    env.reset(task_id="easy-clean-approve")
    assert env.state.opened_targets == []
    assert env.state.logged_findings == []
    assert env.state.draft_decision is None


def test_step_limit_ends_episode() -> None:
    env = VendorOnboardingEnvironment()
    env.reset(task_id="easy-clean-approve")
    final_obs = None
    for _ in range(12):
        final_obs = env.step(VendorOnboardingAction(action_type="inspect", target_id="overview"))

    assert final_obs is not None
    assert final_obs.done is True
    assert env.state.timeout_reached is True
    assert final_obs.metadata["info"]["task_grade"]["final_score"] < 1.0

