import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

SELF_COLLABORATION_ROOT = Path("/installed-agent/self-collaboration")
WORKSPACE = Path("/app")
HISTORY_PATH = Path("/logs/agent/session-history.json")


def main() -> None:
    instruction = os.environ["HARBOR_TASK_INSTRUCTION"]
    sys.path.insert(0, str(SELF_COLLABORATION_ROOT))

    agent_module = importlib.import_module("core.agent")
    config_module = importlib.import_module("core.config")
    repo_tools_module = importlib.import_module("core.repo_tools")

    _initialize_repository()
    structure = repo_tools_module.get_repo_structure(str(WORKSPACE))
    task = f"""## Repository Structure
```
{structure}
```

## Task
{instruction}
"""
    session = agent_module.SelfCollabSession(
        config=config_module.CODER_CONFIG,
        repo_path=str(WORKSPACE),
        max_round=int(os.environ.get("SELF_COLLAB_MAX_ROUNDS", "1")),
        analyst_steps=int(os.environ.get("SELF_COLLAB_ANALYST_STEPS", "10")),
        coder_steps=int(os.environ.get("SELF_COLLAB_CODER_STEPS", "15")),
        verbose=True,
    )
    history, analyst_result, coder_result = session.run(task)
    HISTORY_PATH.write_text(
        json.dumps(
            {
                "history": history,
                "analyst_result": analyst_result,
                "coder_result": coder_result,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _initialize_repository() -> None:
    if (WORKSPACE / ".git").exists():
        return
    subprocess.run(["git", "init", "--quiet"], cwd=WORKSPACE, check=True)
    subprocess.run(
        ["git", "config", "user.email", "harbor@example.invalid"],
        cwd=WORKSPACE,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Harbor"], cwd=WORKSPACE, check=True
    )
    subprocess.run(
        ["git", "commit", "--allow-empty", "--quiet", "-m", "Initial workspace"],
        cwd=WORKSPACE,
        check=True,
    )


if __name__ == "__main__":
    main()
