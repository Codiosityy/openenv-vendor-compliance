"""Async client for the vendor onboarding environment."""

from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import (
        VendorOnboardingAction,
        VendorOnboardingObservation,
        VendorOnboardingState,
    )
except ImportError:  # pragma: no cover - supports direct source imports
    from models import (
        VendorOnboardingAction,
        VendorOnboardingObservation,
        VendorOnboardingState,
    )


class VendorOnboardingEnv(
    EnvClient[
        VendorOnboardingAction,
        VendorOnboardingObservation,
        VendorOnboardingState,
    ]
):
    """WebSocket client for vendor onboarding episodes."""

    def _step_payload(self, action: VendorOnboardingAction) -> dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(
        self, payload: dict[str, Any]
    ) -> StepResult[VendorOnboardingObservation]:
        observation_payload = dict(payload.get("observation", {}))
        observation_payload["reward"] = payload.get("reward")
        observation_payload["done"] = payload.get("done", False)
        observation = VendorOnboardingObservation.model_validate(observation_payload)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> VendorOnboardingState:
        return VendorOnboardingState.model_validate(payload)
