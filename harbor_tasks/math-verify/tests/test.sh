#!/bin/bash
set -uo pipefail

mkdir -p /logs/verifier
output=/logs/verifier/pytest-output.log

for source in /app/* /app/.[!.]* /app/..?*; do
  [ -e "$source" ] || continue
  case "$(basename "$source")" in
    .git|tests|setup.py|pyproject.toml|setup.cfg|requirements.txt|requirements-dev.txt|Pipfile|Pipfile.lock|poetry.lock|tox.ini|Dockerfile)
      continue
      ;;
  esac
  cp -a "$source" /benchmark/
done

cd /benchmark
pip install -e . 2>&1 | tee /logs/verifier/install-output.log
install_status=${PIPESTATUS[0]}

if [ "$install_status" -eq 0 ]; then
  python -m pytest --continue-on-collection-errors -n 0 tests/test_all.py  -v -s 2>&1 | tee "$output"
  test_status=${PIPESTATUS[0]}
else
  test_status=$install_status
  printf 'Project installation failed; tests were not run.\n' | tee "$output"
fi

python - "$output" "$test_status" <<'PY'
import json
import re
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
exit_code = int(sys.argv[2])
output = output_path.read_text(encoding="utf-8", errors="replace")

def count(label: str) -> int:
    matches = re.findall(rf"(\d+) {label}", output)
    return int(matches[-1]) if matches else 0

passed = count("passed")
failed = count("failed")
errors = count("errors?")
total = 192
score = min(passed / total, 1.0) if total else 0.0
result = {
    "passed": passed,
    "failed": failed,
    "errors": errors,
    "total": total,
    "success_rate": score,
    "exit_code": exit_code,
}
Path("/logs/verifier/evaluator-output.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8"
)
Path("/logs/verifier/reward.txt").write_text(f"{score}\n", encoding="utf-8")
PY
