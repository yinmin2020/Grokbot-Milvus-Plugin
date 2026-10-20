"""Real embeddings + real backend. Enable with RUN_EMBEDDING_E2E=1."""

import json
import os
import subprocess
import sys
from pathlib import Path

import milvus_kb as kb
import milvus_memory as memory
import pytest
from mvbot_common import query_rows

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(os.getenv("RUN_EMBEDDING_E2E") != "1", reason="real embedding E2E opt-in"),
]


def test_real_document_ingest_and_retrieval(client, collection, tmp_path, capsys):
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, str(root / "scripts/make_sample_docs.py"), str(tmp_path)],
        check=True,
        capture_output=True,
    )
    (tmp_path / "guide.md").write_text(
        "# ZXQERROR42\n\nZXQERROR42 indicates a full disk. Remove unused files to resolve it."
    )
    (tmp_path / "notes.txt").write_text("RRF combines ranked lists. The default RRFRanker k is 60.")

    def args(*values):
        return kb.build_parser().parse_args([*values, "--collection", collection])

    kb.cmd_ingest(client, args("ingest", str(tmp_path), "--dry-run"))
    assert not client.has_collection(collection)
    kb.cmd_ingest(client, args("ingest", str(tmp_path)))
    assert len(kb._docs(client, collection)) == 4
    kb.cmd_ingest(client, args("ingest", str(tmp_path)))
    assert '"unchanged": 4' in capsys.readouterr().out
    for mode in ("dense", "bm25", "hybrid"):
        query = ["query", "ZXQERROR42 full disk", "--mode", mode, "--top-k", "1", "--json"]
        if mode == "dense":
            query += ["--min-score", "0"]
        kb.cmd_query(client, args(*query))
        result = json.loads(capsys.readouterr().out)
        assert result["found"]
        assert result["results"][0]["source"].endswith("guide.md")
        assert "full disk" in result["results"][0]["text"]
    kb.cmd_query(
        client,
        args(
            "query",
            "full disk",
            "--mode",
            "hybrid",
            "--ranker",
            "weighted",
            "--json",
            "--filter",
            'lang == "en"',
        ),
    )
    result = json.loads(capsys.readouterr().out)
    assert result["results"]
    assert all(r["lang"] == "en" for r in result["results"])
    for question, suffix in [
        ("How do I release collection memory?", ".pdf"),
        ("备份之前需要做什么？", ".docx"),
    ]:
        kb.cmd_query(client, args("query", question, "--top-k", "3", "--json"))
        result = json.loads(capsys.readouterr().out)
        assert any(r["source"].endswith(suffix) for r in result["results"])
    doc = next(d for d in kb._docs(client, collection) if d["source"].endswith("guide.md"))
    kb.cmd_delete_doc(
        client, args("delete-doc", "--doc-id", doc["doc_id"], "--execute", "--confirm", collection)
    )
    assert len(kb._docs(client, collection)) == 3


def test_real_memory_embeddings_and_user_isolation(client, collection, tmp_path, capsys):
    def args(*values):
        return memory.build_parser().parse_args([*values, "--collection", collection])

    memory.cmd_init(client, args("init", "--execute"))
    path = tmp_path / "memory.txt"
    for user, text in [
        ("alice", "I prefer the Vim editor."),
        ("bob", "I prefer the Emacs editor."),
        ("alice", "I prefer the VS Code editor."),
    ]:
        path.write_text(text)
        memory.cmd_remember(
            client,
            args("remember", "--user-id", user, "--memory-id", "editor", "--file", str(path), "--execute"),
        )
    capsys.readouterr()
    assert len(list(query_rows(client, collection, "", ["pk"]))) == 2
    for mode in ("dense", "bm25", "hybrid"):
        memory.cmd_recall(client, args("recall", "preferred editor", "--user-id", "alice", "--mode", mode))
        result = json.loads(capsys.readouterr().out)
        assert len(result["results"]) == 1
        assert result["results"][0]["entity"]["user_id"] == "alice"
        assert "VS Code" in result["results"][0]["entity"]["text"]
    memory.cmd_forget(
        client,
        args("forget", "--user-id", "alice", "--memory-id", "editor", "--execute", "--confirm", collection),
    )
    remaining = list(query_rows(client, collection, "", ["user_id"]))
    assert [r["user_id"] for r in remaining] == ["bob"]
