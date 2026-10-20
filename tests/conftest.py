"""Real Milvus backends, with one owned collection prefix per test run."""

import os
import uuid
from types import SimpleNamespace

import pytest
from mvbot_common import connect


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    uri = os.getenv("TEST_MILVUS_URI") or str(tmp_path_factory.mktemp("milvus") / "test.db")
    c = connect(SimpleNamespace(uri=uri, token=os.getenv("TEST_MILVUS_TOKEN", ""), db="default"))
    yield c
    c.close()


@pytest.fixture
def collection(client):
    name = "grokbot_test_" + uuid.uuid4().hex[:16]
    assert not client.has_collection(name)
    yield name
    if client.has_collection(name):
        client.drop_collection(name)


class ProbeEmbedder:
    """Deterministic vectors only for retry/identity/isolation regression tests."""

    provider = "openai"
    model = "probe"

    def __init__(self, *args, **kwargs):
        pass

    def dim(self):
        return 2

    def embed(self, texts):
        return [[1.0, 0.0] if "ZXQERROR42" in t else [0.0, 1.0] for t in texts]


@pytest.fixture
def probe_embedding(monkeypatch):
    import milvus_kb
    import milvus_memory
    import mvbot_common

    monkeypatch.setattr(milvus_kb, "Embedder", ProbeEmbedder)
    monkeypatch.setattr(milvus_memory, "Embedder", ProbeEmbedder)
    monkeypatch.setattr(mvbot_common, "Embedder", ProbeEmbedder)
    for name in ("EMBED_PROVIDER", "EMBED_MODEL", "EMBED_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
