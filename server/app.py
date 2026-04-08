"""FastAPI app for vendor onboarding environment."""

from __future__ import annotations

import os

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "openenv is required for the vendor onboarding web interface. "
        "Install dependencies with 'pip install -e .[dev]'."
    ) from exc

try:
    from ..models import VendorOnboardingAction, VendorOnboardingObservation
    from .vendor_onboarding_environment import VendorOnboardingEnvironment
except ImportError:
    from models import VendorOnboardingAction, VendorOnboardingObservation
    from server.vendor_onboarding_environment import VendorOnboardingEnvironment


app = create_app(
    VendorOnboardingEnvironment,
    VendorOnboardingAction,
    VendorOnboardingObservation,
    env_name="vendor_onboarding_env",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int | None = None) -> None:
    """Run the environment server directly."""
    import uvicorn

    resolved_port = port if port is not None else int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host=host, port=resolved_port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    if args.port is None:
        main()
    else:
        main(port=args.port)
