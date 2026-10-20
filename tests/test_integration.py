import json
import subprocess
import sys
from pathlib import Path

import milvus_inspect as inspect
import milvus_kb as kb
import pytest
from mvbot_common import check_write, query_rows, validate_vectors


def kb_args(collection, *args):
    return kb.build_parser().parse_args([*args, "--collection", collection])


def initialize_kb(client, collection, probe_embedding):
    kb.cmd_init(client, kb_args(collection, "init", "--embed-model", "probe", "--embed-provider", "openai"))


def rows(client, collection):
    return list(query_rows(client, collection, "", ["pk", "doc_id", "doc_hash", "text"]))


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
    from pymilvus import DataType

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=2)
    indexes = client.prepare_index_params()
    indexes.add_index("vector", index_type="AUTOINDEX", metric_type="COSINE")
    client.create_collection(collection, schema=schema, index_params=indexes)
    with pytest.raises(ValueError, match="compatible"):
        kb.cmd_init(client, kb_args(collection, "init", "--embed-model", "probe"))


def test_mismatched_model_rejected(client, collection, probe_embedding):
    initialize_kb(client, collection, probe_embedding)
    with pytest.raises(ValueError, match="differs"):
        kb.cmd_init(client, kb_args(collection, "init", "--embed-model", "different"))


def test_knowledge_base_write_guards(client, collection, probe_embedding, tmp_path, monkeypatch):
    initialize_kb(client, collection, probe_embedding)
    path = tmp_path / "test.txt"
    path.write_text("test")
    monkeypatch.setenv("KB_PROTECTED_COLLECTIONS", collection)
    with pytest.raises(SystemExit, match="protected"):
        kb.cmd_ingest(client, kb_args(collection, "ingest", str(path)))
    with pytest.raises(SystemExit, match="protected"):
        kb.cmd_delete_doc(
            client, kb_args(collection, "delete-doc", "--doc-id", "any", "--execute", "--confirm", collection)
        )
    monkeypatch.setenv("KB_WRITE_ALLOW_PREFIX", "allowed_")
    with pytest.raises(SystemExit, match="KB_WRITE_ALLOW_PREFIX"):
        check_write("other_collection")


def test_validation_rejects_malformed_vectors():
    for vectors in ([[1, 2], [3]], [[1, float("nan")]], [[True, 2]]):
        with pytest.raises(ValueError):
            validate_vectors(vectors, 2)


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


def test_native_retrieval_scores_filters_and_citations(
    client, collection, probe_embedding, document_files, capsys
):
    initialize_kb(client, collection, probe_embedding)
    kb.cmd_ingest(client, kb_args(collection, "ingest", str(document_files)))
    capsys.readouterr()
    for mode in ("dense", "bm25", "hybrid"):
        kb.cmd_query(
            client, kb_args(collection, "query", "ZXQERROR42", "--mode", mode, "--json", "--top-k", "1")
        )
        result = json.loads(capsys.readouterr().out)
        hit = result["results"][0]
        assert hit["source"] == str(document_files / "guide.md")
        assert hit["heading"] == "Error reference > Disk usage"
        assert hit["cite"] == 1 and hit["doc_id"] and hit["chunk_no"] == 0
        assert "full disk" in hit["text"]
        if mode == "dense":
            assert hit["score"] == pytest.approx(1, abs=1e-5)
    for mode in ("dense", "bm25", "hybrid"):
        # A filter excluding the relevant English document must apply before top-k selection.
        args = kb_args(
            collection, "query", "ZXQERROR42", "--mode", mode, "--json", "--filter", 'lang == "zh"'
        )
        kb.cmd_query(client, args)
        result = json.loads(capsys.readouterr().out)
        assert all(r["lang"] == "zh" for r in result["results"])
        if mode in ("dense", "bm25"):
            assert not result["found"]
    kb.cmd_query(client, kb_args(collection, "query", "CITATIONCHECK42", "--mode", "bm25", "--json"))
    result = json.loads(capsys.readouterr().out)
    assert result["results"][0]["source"].endswith("guide_en.pdf")
    assert result["results"][0]["page"] == 2
    kb.cmd_query(client, kb_args(collection, "query", "文档标识", "--mode", "bm25", "--json"))
    result = json.loads(capsys.readouterr().out)
    assert result["results"][0]["source"].endswith("guide_zh.docx")
    assert result["results"][0]["heading"] == "文档更新"


def test_inspection_does_not_load_or_modify_collections(client, collection, probe_embedding, capsys):
    initialize_kb(client, collection, probe_embedding)
    client.release_collection(collection)
    before = str(client.get_load_state(collection)["state"])
    args = inspect.build_parser().parse_args(["health"])
    capsys.readouterr()
    inspect.cmd_health(client, args)
    assert json.loads(capsys.readouterr().out)["connected"]
    inspect.cmd_collections(client, args)
    assert collection in json.loads(capsys.readouterr().out)["collections"]
    args = inspect.build_parser().parse_args(["describe", collection])
    inspect.cmd_describe(client, args)
    result = json.loads(capsys.readouterr().out)
    assert result["collection"] == collection
    assert {f["name"] for f in result["fields"]} >= {"text", "dense", "sparse"}
    assert str(client.get_load_state(collection)["state"]) == before


def test_cli_reopens_lite_and_works_outside_repo(tmp_path, probe_embedding):
    root = Path(__file__).resolve().parents[1]
    uri = str(tmp_path / "cli.db")
    path = tmp_path / 'quoted "source".txt'
    path.write_text("ZXQERROR42 means disk full")
    # Setup also runs in its own process: Lite's embedded server holds its lock until process exit.
    setup = """
import sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path[:0] = [str(root / "scripts"), str(root / "tests")]
import milvus_kb as kb
from conftest import ProbeEmbedder
kb.Embedder = ProbeEmbedder
args = kb.build_parser().parse_args(["ingest", sys.argv[3], "--collection", "cli", "--uri", sys.argv[2], "--embed-model", "probe"])
c = kb.connect(args)
try:
    kb.cmd_ingest(c, args)
finally:
    c.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", setup, str(root), uri, str(path)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr

    def run(script, *args):
        return subprocess.run(
            [sys.executable, str(root / "scripts" / script), *args, "--uri", uri],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            timeout=120,
        )

    result = run("milvus_kb.py", "query", "ZXQERROR42", "--collection", "cli", "--mode", "bm25", "--json")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["results"][0]["source"] == str(path)
    result = run("milvus_inspect.py", "describe", "cli")
    assert result.returncode == 0, result.stderr
    result = run(
        "milvus_kb.py",
        "delete-doc",
        "--collection",
        "cli",
        "--source",
        str(path),
        "--execute",
        "--confirm",
        "wrong",
    )
    assert result.returncode != 0
    result = run(
        "milvus_kb.py",
        "delete-doc",
        "--collection",
        "cli",
        "--source",
        str(path),
        "--execute",
        "--confirm",
        "cli",
    )
    assert result.returncode == 0, result.stderr
    result = run("milvus_kb.py", "info", "--collection", "cli")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["row_count"] == 0
