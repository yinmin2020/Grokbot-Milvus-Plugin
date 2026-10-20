import json
import subprocess
import sys
from pathlib import Path

import milvus_data as data
import milvus_kb as kb
import milvus_memory as memory
import milvus_ops as ops
import pytest
from mvbot_common import check_write, query_rows, validate_vectors
from mvbot_search import search


def kb_args(collection, *args):
    return kb.build_parser().parse_args([*args, "--collection", collection])


def mem_args(collection, *args):
    return memory.build_parser().parse_args([*args, "--collection", collection])


def initialize_kb(client, collection, probe_embedding):
    kb.cmd_init(client, kb_args(collection, "init", "--embed-model", "probe", "--embed-provider", "openai"))


def rows(client, collection):
    return list(query_rows(client, collection, "", ["pk", "doc_id", "doc_hash", "text"]))


def test_metadata_and_native_scores(client, collection):
    data.create(client, collection, "hybrid", 2)
    assert data.read_config(client, collection)["dim"] == 2
    client.upsert(
        collection,
        [
            {"pk": "same", "text": "ZXQERROR42 blue bird", "metadata": {"lang": "en"}, "dense": [1.0, 0.0]},
            {"pk": "orthogonal", "text": "其他文字", "metadata": {"lang": "zh"}, "dense": [0.0, 1.0]},
        ],
    )
    hits = search(client, collection, mode="dense", vector=[1, 0], top_k=2)
    assert hits[0]["pk"] == "same"
    assert hits[0]["distance"] == pytest.approx(1, abs=1e-5)
    assert hits[1]["distance"] == pytest.approx(0, abs=1e-5)
    for mode in ("bm25", "hybrid"):
        hits = search(
            client,
            collection,
            mode=mode,
            text="ZXQERROR42",
            vector=[1, 0],
            filter='metadata["lang"] == "en"',
            top_k=1,
        )
        assert [hit["pk"] for hit in hits] == ["same"]
    hits = search(client, collection, mode="hybrid", text="ZXQERROR42", vector=[1, 0], ranker="weighted")
    assert hits[0]["pk"] == "same"


def test_document_identity_and_working_directory(client, collection, probe_embedding, tmp_path, monkeypatch):
    initialize_kb(client, collection, probe_embedding)
    first = tmp_path / "project_a" / "docs" / "readme.md"
    second = tmp_path / "project_b" / "docs" / "readme.md"
    for path, text in [(first, "Project A apples"), (second, "Project B bananas")]:
        path.parent.mkdir(parents=True)
        path.write_text(text)
        monkeypatch.chdir(path.parent.parent)
        kb.cmd_ingest(client, kb_args(collection, "ingest", "docs/readme.md"))
    assert len({r["doc_id"] for r in rows(client, collection)}) == 2
    monkeypatch.chdir(tmp_path)
    kb.cmd_ingest(client, kb_args(collection, "ingest", str(first)))
    assert len(rows(client, collection)) == 2
    a = kb_args(
        collection, "ingest", str(first), "--namespace", "project-a", "--source-root", str(first.parent)
    )
    b = kb_args(
        collection, "ingest", str(second), "--namespace", "project-a", "--source-root", str(second.parent)
    )
    assert kb.source_identity(first, a)[1] == kb.source_identity(second, b)[1]
    b.namespace = "project-b"
    assert kb.source_identity(first, a)[1] != kb.source_identity(second, b)[1]


def test_retry_completes_partial_batches_and_removes_stale(
    client, collection, probe_embedding, tmp_path, monkeypatch, capsys
):
    initialize_kb(client, collection, probe_embedding)
    path = tmp_path / "large.md"
    path.write_text("\n\n".join(f"## Section {i}\n\n" + "word " * 20 for i in range(130)))
    args = kb_args(collection, "ingest", str(path))
    original = client.upsert
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ConnectionError("injected second-batch failure")
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(client, "upsert", fail_second)
        with pytest.raises(ConnectionError):
            kb.cmd_ingest(client, args)
    assert len(rows(client, collection)) == 64
    kb.cmd_ingest(client, args)
    assert len(rows(client, collection)) == 130
    capsys.readouterr()
    kb.cmd_ingest(client, args)
    assert '"unchanged": 1' in capsys.readouterr().out
    path.write_text("A much shorter replacement")
    kb.cmd_ingest(client, args)
    assert len(rows(client, collection)) == 1
    assert rows(client, collection)[0]["text"] == "A much shorter replacement"


def test_empty_update_reports_failure_and_retains_data(client, collection, probe_embedding, tmp_path):
    initialize_kb(client, collection, probe_embedding)
    path = tmp_path / "note.txt"
    path.write_text("Keep this until explicitly deleted")
    args = kb_args(collection, "ingest", str(path))
    kb.cmd_ingest(client, args)
    path.write_text("")
    with pytest.raises(ValueError, match="existing chunks retained"):
        kb.cmd_ingest(client, args)
    assert len(rows(client, collection)) == 1


def test_bm25_does_not_call_embedding_or_apply_dense_gate(
    client, collection, probe_embedding, tmp_path, monkeypatch, capsys
):
    initialize_kb(client, collection, probe_embedding)
    path = tmp_path / "keyword.txt"
    path.write_text("ZXQERROR42 means disk full")
    kb.cmd_ingest(client, kb_args(collection, "ingest", str(path)))

    def forbidden(*args, **kwargs):
        pytest.fail("BM25 must not resolve an embedder")

    monkeypatch.setattr(kb, "resolve_embedding", forbidden)
    capsys.readouterr()
    kb.cmd_query(client, kb_args(collection, "query", "ZXQERROR42", "--mode", "bm25", "--json"))
    result = json.loads(capsys.readouterr().out)
    assert result["found"]
    assert "ZXQERROR42" in result["results"][0]["text"]
    with pytest.raises(ValueError, match="dense only"):
        kb.cmd_query(
            client, kb_args(collection, "query", "ZXQERROR42", "--mode", "bm25", "--min-score", "0.9")
        )


def test_foreign_schema_and_model_rejected(client, collection, probe_embedding):
    data.create(client, collection, "dense", 2)
    with pytest.raises(ValueError, match="compatible"):
        kb.cmd_init(client, kb_args(collection, "init", "--embed-model", "probe"))


def test_mismatched_model_rejected(client, collection, probe_embedding):
    initialize_kb(client, collection, probe_embedding)
    with pytest.raises(ValueError, match="differs"):
        kb.cmd_init(client, kb_args(collection, "init", "--embed-model", "different"))


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_memory_scope_before_topk_and_escaped_user_ids(
    client, collection, probe_embedding, tmp_path, capsys, mode
):
    memory.cmd_init(client, mem_args(collection, "init", "--embed-model", "probe", "--execute"))
    victim = 'user "one"\\path'
    users = ["another-user"] * 6 + [victim]
    path = tmp_path / "memory.txt"
    path.write_text("ZXQERROR42 is my preference")
    for i, user in enumerate(users):
        memory.cmd_remember(
            client,
            mem_args(
                collection,
                "remember",
                "--user-id",
                user,
                "--memory-id",
                str(i) if user != victim else "0",
                "--file",
                str(path),
                "--execute",
            ),
        )
    capsys.readouterr()
    memory.cmd_recall(
        client,
        mem_args(collection, "recall", "ZXQERROR42", "--user-id", victim, "--top-k", "1", "--mode", mode),
    )
    result = json.loads(capsys.readouterr().out)
    assert len(result["results"]) == 1
    assert result["results"][0]["entity"]["user_id"] == victim
    memory.cmd_forget(
        client,
        mem_args(
            collection,
            "forget",
            "--user-id",
            victim,
            "--memory-id",
            "0",
            "--execute",
            "--confirm",
            collection,
        ),
    )
    remaining = list(query_rows(client, collection, "", ["user_id"]))
    assert len(remaining) == 6
    assert all(r["user_id"] == "another-user" for r in remaining)


def test_memory_native_upsert_preserves_other_user(client, collection, probe_embedding, tmp_path):
    memory.cmd_init(client, mem_args(collection, "init", "--embed-model", "probe", "--execute"))
    path = tmp_path / "memory.txt"
    for user, content in [("a", "old"), ("b", "private"), ("a", "updated")]:
        path.write_text(content)
        memory.cmd_remember(
            client,
            mem_args(
                collection,
                "remember",
                "--user-id",
                user,
                "--memory-id",
                "same",
                "--file",
                str(path),
                "--execute",
            ),
        )
    found = list(query_rows(client, collection, "", ["user_id", "text"]))
    assert {r["user_id"]: r["text"] for r in found} == {"a": "updated", "b": "private"}


def test_write_guard_shared_by_all_scripts(client, collection, probe_embedding, tmp_path, monkeypatch):
    initialize_kb(client, collection, probe_embedding)
    path = tmp_path / "test.txt"
    path.write_text("test")
    monkeypatch.setenv("OPS_PROTECTED_COLLECTIONS", collection)
    with pytest.raises(SystemExit, match="protected"):
        kb.cmd_ingest(client, kb_args(collection, "ingest", str(path)))
    with pytest.raises(SystemExit, match="protected"):
        kb.cmd_delete_doc(
            client, kb_args(collection, "delete-doc", "--doc-id", "any", "--execute", "--confirm", collection)
        )
    with pytest.raises(SystemExit, match="protected"):
        ops.guard(
            client,
            ops.build_parser().parse_args(
                ["drop-collection", collection, "--execute", "--confirm", collection]
            ),
            "drop",
        )
    with pytest.raises(SystemExit, match="protected"):
        data.cmd_create(
            client, data.build_parser().parse_args(["create", "--collection", collection, "--execute"])
        )
    with pytest.raises(SystemExit, match="protected"):
        memory.cmd_init(client, mem_args(collection, "init", "--execute"))
    monkeypatch.setenv("OPS_WRITE_ALLOW_PREFIX", "allowed_")
    with pytest.raises(SystemExit, match="OPS_WRITE_ALLOW_PREFIX"):
        check_write("other_collection")


def test_validation_rejects_malformed_vectors():
    for vectors in ([[1, 2], [3]], [[1, float("nan")]], [[True, 2]]):
        with pytest.raises(ValueError):
            validate_vectors(vectors, 2)


def test_cli_reopens_lite_and_works_outside_repo(tmp_path):
    root = Path(__file__).resolve().parents[1]
    uri = str(tmp_path / "cli.db")

    def run(script, *args):
        return subprocess.run(
            [sys.executable, str(root / "scripts" / script), *args, "--uri", uri],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            timeout=120,
        )

    result = run("milvus_data.py", "create", "--collection", "cli", "--mode", "bm25", "--execute")
    assert result.returncode == 0, result.stderr
    path = tmp_path / "rows.json"
    path.write_text(json.dumps([{"pk": "one", "text": "ZXQERROR42"}]))
    result = run("milvus_data.py", "upsert", "--collection", "cli", "--file", str(path), "--execute")
    assert result.returncode == 0, result.stderr
    result = run("milvus_data.py", "search", "--collection", "cli", "--mode", "bm25", "--text", "ZXQERROR42")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["results"][0]["pk"] == "one"
    result = run("milvus_ops.py", "drop-collection", "cli", "--execute", "--confirm", "wrong")
    assert result.returncode != 0


@pytest.mark.parametrize("metric,expected", [("COSINE", [1, 0]), ("IP", [1, 0]), ("L2", [0, 2])])
def test_metric_ordering_on_native_backend(client, collection, metric, expected):
    from pymilvus import DataType

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("pk", DataType.VARCHAR, is_primary=True, max_length=20)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=2)
    indexes = client.prepare_index_params()
    indexes.add_index("dense", index_type="AUTOINDEX", metric_type=metric)
    client.create_collection(collection, schema=schema, index_params=indexes)
    client.upsert(collection, [{"pk": "same", "dense": [1, 0]}, {"pk": "orthogonal", "dense": [0, 1]}])
    result = search(
        client, collection, mode="dense", vector=[1, 0], metric=metric, top_k=2, output_fields=["pk"]
    )
    assert [r["pk"] for r in result] == ["same", "orthogonal"]
    assert [r["distance"] for r in result] == pytest.approx(expected, abs=1e-5)


def test_validator_failure_exit_status(tmp_path):
    import shutil

    root = Path(__file__).resolve().parents[1]
    for folder in ("scripts", "skills", ".cursor-plugin", "assets"):
        shutil.copytree(root / folder, tmp_path / folder, ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("README.md", "pyproject.toml", "uv.lock"):
        shutil.copy2(root / name, tmp_path / name)
    manifest = tmp_path / ".cursor-plugin/plugin.json"
    content = json.loads(manifest.read_text())
    content["version"] = "invalid"
    manifest.write_text(json.dumps(content))
    result = subprocess.run(
        [sys.executable, str(tmp_path / "scripts/validate_plugin.py"), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert "INVALID" in result.stdout


def test_destructive_confirmation_preserves_data(client, collection, tmp_path, probe_embedding):
    initialize_kb(client, collection, probe_embedding)
    path = tmp_path / "keep.txt"
    path.write_text("This content must survive an unconfirmed delete")
    kb.cmd_ingest(client, kb_args(collection, "ingest", str(path)))
    doc_id = rows(client, collection)[0]["doc_id"]
    for flags in ([], ["--execute", "--confirm", "wrong"]):
        args = kb_args(collection, "delete-doc", "--doc-id", doc_id, *flags)
        if flags:
            with pytest.raises(SystemExit, match="REFUSED"):
                kb.cmd_delete_doc(client, args)
        else:
            kb.cmd_delete_doc(client, args)
        assert len(rows(client, collection)) == 1


def test_server_maintenance(client, collection, capsys):
    import os

    if not os.getenv("TEST_MILVUS_URI", "").startswith(("http://", "https://")):
        pytest.skip("Server administration is not available on Lite")
    data.create(client, collection, "dense", 2)
    client.upsert(collection, [{"pk": "one", "text": "record", "metadata": {}, "dense": [1, 0]}])

    def run(*args):
        a = ops.build_parser().parse_args(args)
        globals_fn = getattr(ops, "cmd_" + a.cmd.replace("-", "_"))
        globals_fn(client, a)

    run("describe", collection)
    run("partitions", collection)
    run("segments", collection)
    run("partition-create", collection, "archive", "--execute")
    assert client.has_partition(collection, "archive")
    run("flush", collection, "--execute")
    run("index-create", collection, "--field", "text", "--index-type", "INVERTED", "--execute")
    run("load", collection, "--execute")
    run("delete-entities", collection, "--filter", 'pk == "one"')
    assert len(list(query_rows(client, collection, "", ["pk"]))) == 1
    run("delete-entities", collection, "--filter", 'pk == "one"', "--execute", "--confirm", collection)
    assert list(query_rows(client, collection, "", ["pk"])) == []
    run("release", collection, "--execute", "--confirm", collection)
    run("index-drop", collection, "--index", "text", "--execute", "--confirm", collection)
    run("partition-drop", collection, "archive", "--execute", "--confirm", collection)
    assert not client.has_partition(collection, "archive")
    run("drop-collection", collection, "--execute", "--confirm", collection)
    assert not client.has_collection(collection)
