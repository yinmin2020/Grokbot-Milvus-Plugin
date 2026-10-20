"""Run tests with a real embedding endpoint. Configure EMBED_* and optional TEST_MILVUS_* variables."""

import os
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_e2e.py", "-v"],
        cwd=root,
        env={**os.environ, "RUN_EMBEDDING_E2E": "1"},
    )
    sys.exit(result.returncode)
