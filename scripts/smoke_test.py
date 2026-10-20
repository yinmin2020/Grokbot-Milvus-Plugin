"""Run a keyword-only CLI smoke test against a fresh Lite database, without external services."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="milvus_plugin_smoke_") as tmp:
        work = Path(tmp)
        uri = str(work / "smoke.db")

        def run(script, *args):
            result = subprocess.run(
                [sys.executable, str(here / script), *args, "--uri", uri],
                cwd=work,
                text=True,
                capture_output=True,
                timeout=120,
            )
            if result.returncode:
                raise RuntimeError(result.stderr)
            return result.stdout

        run("milvus_data.py", "create", "--collection", "smoke", "--mode", "bm25", "--execute")
        path = work / "rows.json"
        path.write_text(
            json.dumps([{"pk": "one", "text": "Milvus supports BM25 search", "metadata": {"lang": "en"}}])
        )
        run("milvus_data.py", "upsert", "--collection", "smoke", "--file", str(path), "--execute")
        found = json.loads(
            run(
                "milvus_data.py",
                "search",
                "--collection",
                "smoke",
                "--mode",
                "bm25",
                "--text",
                "BM25",
                "--filter",
                'metadata["lang"] == "en"',
            )
        )
        assert found["results"][0]["pk"] == "one"
        run("milvus_ops.py", "drop-collection", "smoke", "--execute", "--confirm", "smoke")
        print("PASS: create, upsert, filtered BM25, reopen Lite, guarded drop")


if __name__ == "__main__":
    main()
