"""Explicit schema creation, native upsert, query and filtered search via MilvusClient."""

import argparse
import json
import sys
from pathlib import Path

from mvbot_common import (
    add_milvus_args,
    checked_text,
    collection_meta,
    confirm_write,
    connect,
    ensure_loaded,
    print_json,
    validate_schema,
    validate_vectors,
)
from mvbot_search import search

DATA_TAG = "mvbot-data/v1"
MEMORY_TAG = "mvbot-memory/v1"
ANALYZER = {"tokenizer": "jieba", "filter": ["lowercase", "cnalphanumonly"]}


def create(c, collection, mode, dim=None, memory=False, config=None):
    from pymilvus import DataType, Function, FunctionType

    if mode != "bm25" and (not isinstance(dim, int) or dim < 2):
        raise ValueError("Dense/hybrid collections require --dim >= 2")
    if c.has_collection(collection):
        raise ValueError(f"Collection '{collection}' already exists; inspect it before reuse")
    meta = {"schema": MEMORY_TAG if memory else DATA_TAG, "mode": mode, "dim": dim, **(config or {})}
    schema = c.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("pk", DataType.VARCHAR, is_primary=True, max_length=256)
    schema.add_field(
        "text",
        DataType.VARCHAR,
        max_length=65535,
        enable_analyzer=mode != "dense",
        analyzer_params=ANALYZER if mode != "dense" else {},
    )
    schema.add_field("metadata", DataType.JSON)
    indexes = c.prepare_index_params()
    if memory:
        schema.add_field("user_id", DataType.VARCHAR, max_length=256, is_partition_key=True)
        schema.add_field("memory_id", DataType.VARCHAR, max_length=256)
        indexes.add_index("user_id", index_type="INVERTED")
    if mode != "bm25":
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=dim)
        indexes.add_index("dense", index_type="AUTOINDEX", metric_type="COSINE")
    if mode != "dense":
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(
            Function(
                name="text_bm25",
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names=["sparse"],
            )
        )
        indexes.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    c.create_collection(
        collection,
        schema=schema,
        index_params=indexes,
        consistency_level="Strong",
        properties={"mvbot.config": json.dumps(meta)},
    )
    return meta


def read_config(c, collection, memory=False):
    from pymilvus import DataType

    schema, meta = collection_meta(c, collection, MEMORY_TAG if memory else DATA_TAG)
    fields = {"pk": DataType.VARCHAR, "text": DataType.VARCHAR, "metadata": DataType.JSON}
    if memory:
        fields.update(user_id=DataType.VARCHAR, memory_id=DataType.VARCHAR)
    if meta["mode"] != "bm25":
        fields["dense"] = DataType.FLOAT_VECTOR
    if meta["mode"] != "dense":
        fields["sparse"] = DataType.SPARSE_FLOAT_VECTOR
        if not any(
            f.get("input_field_names") == ["text"] and f.get("output_field_names") == ["sparse"]
            for f in schema.get("functions", [])
        ):
            raise ValueError("Expected text-to-sparse BM25 function")
    validate_schema(schema, fields, meta["dim"] if meta["mode"] != "bm25" else None)
    return meta


def cmd_create(c, a):
    if confirm_write(a):
        print_json(create(c, a.collection, a.mode, a.dim))


def cmd_upsert(c, a):
    meta = read_config(c, a.collection)
    rows = json.loads(Path(a.file).read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("--file must contain a nonempty JSON array")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each row must be a JSON object")
        checked_text(row.get("pk"), "pk", 256)
        if row["pk"] in seen:
            raise ValueError("Duplicate primary keys in one input file")
        seen.add(row["pk"])
        checked_text(row.get("text"), "text", 65535)
        row.setdefault("metadata", {})
        if not isinstance(row["metadata"], dict):
            raise ValueError("metadata must be a JSON object")
        allowed = {"pk", "text", "metadata"}
        if meta["mode"] != "bm25":
            allowed.add("dense")
            validate_vectors([row.get("dense", [])], meta["dim"])
        if set(row) - allowed:
            raise ValueError(f"Unexpected fields: {set(row) - allowed}")
    if not confirm_write(a):
        return
    for i in range(0, len(rows), 64):
        c.upsert(a.collection, rows[i : i + 64])
    print_json({"upserted": len(rows), "collection": a.collection})


def cmd_query(c, a):
    if not 1 <= a.limit <= 16384:
        raise ValueError("Require 1 <= limit <= 16384")
    ensure_loaded(c, a.collection)
    print_json(
        c.query(
            a.collection,
            filter=a.filter,
            output_fields=a.output_fields.split(","),
            limit=a.limit,
            consistency_level="Strong",
        )
    )


def cmd_search(c, a):
    vector = json.loads(Path(a.vector_file).read_text()) if a.vector_file else None
    if a.mode != "bm25":
        fields = {f["name"]: f for f in c.describe_collection(a.collection)["fields"]}
        if a.dense_field not in fields:
            raise ValueError("Dense field does not exist")
        validate_vectors([vector or []], int(fields[a.dense_field]["params"]["dim"]))
    hits = search(
        c,
        a.collection,
        mode=a.mode,
        text=a.text,
        vector=vector,
        filter=a.filter,
        top_k=a.top_k,
        candidates=max(a.top_k, a.candidates),
        ranker=a.ranker,
        weights=tuple(float(x) for x in a.weights.split(",")),
        output_fields=a.output_fields.split(","),
        dense_field=a.dense_field,
        sparse_field=a.sparse_field,
        metric=a.metric,
    )
    print_json({"mode": a.mode, "results": hits})


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("create", "upsert", "query", "search"):
        p = sub.add_parser(name)
        add_milvus_args(p)
        p.add_argument("--collection", required=True)
        if name in ("create", "upsert"):
            p.add_argument("--execute", action="store_true")
        if name == "create":
            p.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid")
            p.add_argument("--dim", type=int)
        if name == "upsert":
            p.add_argument("--file", required=True)
        if name in ("query", "search"):
            p.add_argument("--filter", default="")
            p.add_argument("--output-fields", default="pk,text,metadata")
        if name == "query":
            p.add_argument("--limit", type=int, default=20)
        if name == "search":
            p.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid")
            p.add_argument("--text")
            p.add_argument("--vector-file", help="JSON array containing one query vector")
            p.add_argument("--dense-field", default="dense")
            p.add_argument("--sparse-field", default="sparse")
            p.add_argument("--metric", choices=["COSINE", "IP", "L2"], default="COSINE")
            p.add_argument("--top-k", type=int, default=5)
            p.add_argument("--candidates", type=int, default=20)
            p.add_argument("--ranker", choices=["rrf", "weighted"], default="rrf")
            p.add_argument("--weights", default="0.7,0.3")
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
