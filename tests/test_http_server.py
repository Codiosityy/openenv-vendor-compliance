"""HTTP smoke tests for the FastAPI app."""

from fastapi.testclient import TestClient

from vendor_onboarding_env.server.app import app


def test_http_endpoints_smoke() -> None:
    client = TestClient(app)

    health_response = client.get("/health")
    assert health_response.status_code == 200
    assert health_response.json()["status"] == "healthy"

    reset_response = client.post("/reset", json={"task_id": "easy-clean-approve"})
    assert reset_response.status_code == 200
    assert reset_response.json()["done"] is False

    step_response = client.post(
        "/step",
        json={"action": {"action_type": "inspect", "target_id": "overview"}},
    )
    assert step_response.status_code == 200
    assert step_response.json()["observation"]["current_target_id"] == "overview"

    state_response = client.get("/state")
    assert state_response.status_code == 200
    assert "step_count" in state_response.json()
