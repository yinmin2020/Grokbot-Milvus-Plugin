"""Real embedding API and database validation. Enable with RUN_EMBEDDING_E2E=1."""

import json
import os

import milvus_kb as kb
import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(os.getenv("RUN_EMBEDDING_E2E") != "1", reason="real embedding E2E opt-in"),
]


def test_document_lifecycle_with_real_embeddings(client, collection, document_files, capsys):
    def args(*values):
        return kb.build_parser().parse_args([*values, "--collection", collection])

    kb.cmd_ingest(client, args("ingest", str(document_files), "--dry-run"))
    assert not client.has_collection(collection)
    kb.cmd_ingest(client, args("ingest", str(document_files)))
    assert len(kb._docs(client, collection)) == 4
    kb.cmd_ingest(client, args("ingest", str(document_files)))
    assert '"unchanged": 4' in capsys.readouterr().out
    for mode in ("dense", "bm25", "hybrid"):
        query = ["query", "ZXQERROR42 full disk", "--mode", mode, "--top-k", "1", "--json"]
        if mode == "dense":
            query += ["--min-score", "0"]
        kb.cmd_query(client, args(*query))
        result = json.loads(capsys.readouterr().out)
        assert result["found"]
        hit = result["results"][0]
        assert hit["source"].endswith("guide.md")
        assert hit["heading"] == "Error reference > Disk usage"
        assert hit["cite"] == 1 and "full disk" in hit["text"]
    kb.cmd_query(
        client, args("query", "full disk", "--ranker", "weighted", "--json", "--filter", 'lang == "en"')
    )
    result = json.loads(capsys.readouterr().out)
    assert result["results"] and all(r["lang"] == "en" for r in result["results"])
    for question, filename, page, heading in [
        ("What does CITATIONCHECK42 link to?", "guide_en.pdf", 2, ""),
        ("修改文档后如何更新知识库？", "guide_zh.docx", 0, "文档更新"),
    ]:
        kb.cmd_query(client, args("query", question, "--top-k", "1", "--json"))
        hit = json.loads(capsys.readouterr().out)["results"][0]
        assert hit["source"].endswith(filename)
        assert hit["page"] == page and hit["heading"] == heading
    original = next(d for d in kb._docs(client, collection) if d["source"].endswith("guide.md"))
    (document_files / "guide.md").write_text(
        "# Updated reference\n\nZXQERROR42 now requires increasing the disk quota."
    )
    kb.cmd_ingest(client, args("ingest", str(document_files)))
    capsys.readouterr()
    kb.cmd_query(client, args("query", "ZXQERROR42", "--mode", "bm25", "--json"))
    hit = json.loads(capsys.readouterr().out)["results"][0]
    assert hit["doc_id"] == original["doc_id"]
    assert "increasing the disk quota" in hit["text"] and "Remove unused" not in hit["text"]
    assert hit["heading"] == "Updated reference"
    kb.cmd_delete_doc(
        client, args("delete-doc", "--doc-id", hit["doc_id"], "--execute", "--confirm", collection)
    )
    capsys.readouterr()
    kb.cmd_query(client, args("query", "ZXQERROR42", "--mode", "bm25", "--json"))
    assert not json.loads(capsys.readouterr().out)["found"]
    assert len(kb._docs(client, collection)) == 3
