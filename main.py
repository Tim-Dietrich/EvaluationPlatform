import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).parent


def build_harbor_config(environment: Mapping[str, str]) -> dict[str, Any]:
    config = yaml.safe_load((ROOT / "experiment.yaml").read_text(encoding="utf-8"))
    provider = environment.get("MODEL_PROVIDER", "openrouter")
    model = environment.get("MODEL", "moonshotai/kimi-k2.5")
    config["agents"][0]["model_name"] = f"{provider}/{model}"
    return config


def build_harbor_command(
        config_path: Path,
        python_executable: str = sys.executable,
) -> list[str]:
    executable_name = "harbor.exe" if os.name == "nt" else "harbor"
    harbor_executable = str(Path(python_executable).with_name(executable_name))
    return [
        harbor_executable,
        "run",
        "--config",
        str(config_path),
    ]


def main() -> int:
    load_dotenv(ROOT / ".env")
    jobs_dir = ROOT / "jobs"
    jobs_dir.mkdir(exist_ok=True)
    config_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".yaml",
                prefix="experiment-",
                dir=jobs_dir,
                delete=False,
        ) as config_file:
            yaml.safe_dump(build_harbor_config(os.environ), config_file, sort_keys=False)
            config_path = Path(config_file.name)
        result = subprocess.run(build_harbor_command(config_path), check=False)
        return result.returncode
    finally:
        if config_path is not None:
            config_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
