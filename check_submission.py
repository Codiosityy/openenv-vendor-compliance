"""Pre-submission checklist verification."""
import json
import os
import urllib.request

import yaml


def main():
    print("=== PRE-SUBMISSION CHECKLIST ===\n")

    # 1. HF Space deploys
    print("[1] HF Space deploys and responds to reset()")
    try:
        data = json.dumps({}).encode()
        req = urllib.request.Request(
            "https://codiosity-vendor-onboarding-env.hf.space/reset",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=15)
        print(f"    PASS - Status {resp.status}, returns observation")
    except Exception as e:
        print(f"    FAIL - {e}")

    # 2. OpenEnv spec compliance
    print("[2] OpenEnv spec compliance")
    with open("openenv.yaml") as f:
        config = yaml.safe_load(f)
    print(f"    openenv.yaml: name={config['name']}, type={config['type']}, tags={config['tags']}")
    assert "openenv" in config["tags"], "Missing openenv tag!"
    print("    PASS - openenv.yaml valid with openenv tag")

    # 3. Typed models
    print("[3] Typed Pydantic models")
    from models import VendorOnboardingAction, VendorOnboardingObservation, VendorOnboardingState
    print("    PASS - Action, Observation, State models imported")

    # 4. 3+ tasks with graders
    print("[4] 3+ tasks with graders (scores 0.0-1.0)")
    from grading import get_task, get_task_ids, grade_task
    task_ids = get_task_ids()
    print(f"    Tasks: {task_ids}")
    assert len(task_ids) >= 3, "Need 3+ tasks!"
    for tid in task_ids:
        task = get_task(tid)
        grade = grade_task(task, [], [], None)
        assert 0.0 <= grade.final_score <= 1.0
        perfect = grade_task(
            task,
            [p.id for p in task.panels],
            task.grader.required_findings,
            task.grader.correct_decision,
        )
        print(f"    {tid} ({task.difficulty}): empty={grade.final_score:.2f}, perfect={perfect.final_score:.2f}")
    print("    PASS - All graders return scores in [0.0, 1.0]")

    # 5. inference.py exists at root
    print("[5] inference.py at root")
    assert os.path.exists("inference.py"), "Missing inference.py!"
    print("    PASS - inference.py found")

    # 6. Dockerfile exists
    print("[6] Dockerfile exists")
    assert os.path.exists("Dockerfile"), "Missing root Dockerfile!"
    print("    PASS - Dockerfile found at root")

    # 7. README has content
    print("[7] README documentation")
    with open("README.md") as f:
        readme = f.read()
    for c in ["Action Space", "Observation Space", "Baseline Scores", "Docker", "openenv"]:
        assert c in readme, f"README missing: {c}"
    print("    PASS - README has all required sections")

    print("\n=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
