"""Model validation tests."""

from pydantic import ValidationError

from vendor_onboarding_env import VendorOnboardingAction


def test_inspect_requires_target_id() -> None:
    try:
        VendorOnboardingAction(action_type="inspect")
    except ValidationError as exc:
        assert "target_id" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected inspect action validation to fail")


def test_submit_rejects_extra_fields() -> None:
    try:
        VendorOnboardingAction(action_type="submit", decision="approve")
    except ValidationError as exc:
        assert "do not accept" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected submit action validation to fail")

