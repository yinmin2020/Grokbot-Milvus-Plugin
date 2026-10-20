"""Explicit user-scoped remember, recall, list and forget operations via MilvusClient."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from milvus_data import create, read_config
from mvbot_common import (
    Embedder,
    add_embed_args,
    add_milvus_args,
    checked_text,
    confirm_write,
    connect,
    default_prefixes,
    ensure_loaded,
    literal,
    print_json,
    resolve_embedding,
    validate_vectors,
)
from mvbot_search import search


def scope(user_id):
    checked_text(user_id, "user_id", 256)
    return f"user_id == {literal(user_id)}"


def memory_pk(user_id, memory_id):
    checked_text(user_id, "user_id", 256)
    checked_text(memory_id, "memory_id", 256)
    return hashlib.sha256(json.dumps([user_id, memory_id]).encode()).hexdigest()


def cmd_init(c, a):
    if not confirm_write(a):
        return
    emb = Embedder(a.embed_provider, a.embed_endpoint, a.embed_model, a.embed_batch, a.embed_timeout)
    prefixes = default_prefixes(emb.model)
    meta = create(
        c,
        a.collection,
        "hybrid",
        emb.dim(),
        memory=True,
        config={
            "embed_provider": emb.provider,
            "embed_model": emb.model,
            "query_prefix": prefixes["query"],
            "doc_prefix": prefixes["doc"],
        },
    )
    print_json(meta)


def cmd_remember(c, a):
    meta = read_config(c, a.collection, memory=True)
    scope(a.user_id)
    pk = memory_pk(a.user_id, a.memory_id)
    text = Path(a.file).read_text(encoding="utf-8")
    checked_text(text, "memory text", 65535)
    metadata = json.loads(a.metadata)
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    if not confirm_write(a):
        return
    emb = resolve_embedding(a, meta)
    vector = emb.embed([meta["doc_prefix"].replace("{title}", "memory") + text])[0]
    validate_vectors([vector], meta["dim"])
    c.upsert(
        a.collection,
        [
            {
                "pk": pk,
                "user_id": a.user_id,
                "memory_id": a.memory_id,
                "text": text,
                "metadata": metadata,
                "dense": vector,
            }
        ],
    )
    print_json({"remembered": a.memory_id, "user_id": a.user_id})


def cmd_recall(c, a):
    meta = read_config(c, a.collection, memory=True)
    flt = scope(a.user_id)
    vector = None
    if a.mode != "bm25":
        emb = resolve_embedding(a, meta)
        vector = emb.embed([meta["query_prefix"] + a.question])[0]
        validate_vectors([vector], meta["dim"])
    hits = search(
        c,
        a.collection,
        mode=a.mode,
        text=a.question,
        vector=vector,
        filter=flt,
        top_k=a.top_k,
        candidates=max(20, a.top_k),
        output_fields=["memory_id", "user_id", "text", "metadata"],
    )
    print_json({"user_id": a.user_id, "mode": a.mode, "results": hits})


def cmd_list(c, a):
    read_config(c, a.collection, memory=True)
    if not 1 <= a.limit <= 16384:
        raise ValueError("Require 1 <= limit <= 16384")
    ensure_loaded(c, a.collection)
    print_json(
        c.query(
            a.collection,
            filter=scope(a.user_id),
            limit=a.limit,
            output_fields=["memory_id", "user_id", "text", "metadata"],
            consistency_level="Strong",
        )
    )


def cmd_forget(c, a):
    read_config(c, a.collection, memory=True)
    flt = scope(a.user_id) + f" and pk == {literal(memory_pk(a.user_id, a.memory_id))}"
    if confirm_write(a, destructive=True):
        print_json(c.delete(a.collection, filter=flt))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("init", "remember", "recall", "list", "forget"):
        p = sub.add_parser(name)
        add_milvus_args(p)
        p.add_argument("--collection", default="agent_memory")
        if name != "init":
            p.add_argument("--user-id", required=True)
        if name in ("init", "remember", "recall"):
            add_embed_args(p)
        if name in ("init", "remember", "forget"):
            p.add_argument("--execute", action="store_true")
        if name in ("remember", "forget"):
            p.add_argument("--memory-id", required=True)
        if name == "remember":
            p.add_argument("--file", required=True, help="UTF-8 text file")
            p.add_argument("--metadata", default="{}", help="JSON object")
        if name == "recall":
            p.add_argument("question")
            p.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid")
            p.add_argument("--top-k", type=int, default=5)
        if name == "list":
            p.add_argument("--limit", type=int, default=20)
        if name == "forget":
            p.add_argument("--confirm", default="")
    return parser


def main():
    a = build_parser().parse_args()
    c = connect(a)
    try:
        globals()["cmd_" + a.cmd](c, a)
    finally:
        c.close()


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
