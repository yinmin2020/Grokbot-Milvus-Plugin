"""Read-only connection checks and collection inspection through MilvusClient."""

import argparse
import sys
import time

from mvbot_common import add_milvus_args, connect, print_json, redact_uri


def cmd_health(c, a):
    started = time.monotonic()
    collections = c.list_collections(timeout=a.timeout)
    print_json(
        {
            "connected": True,
            "uri": redact_uri(a.uri),
            "database": a.db,
            "collection_count": len(collections),
            "rpc_ms": round((time.monotonic() - started) * 1000, 1),
        }
    )


def cmd_collections(c, a):
    print_json({"database": a.db, "collections": sorted(c.list_collections(timeout=a.timeout))})


def cmd_describe(c, a):
    from pymilvus import DataType

    schema = c.describe_collection(a.collection, timeout=a.timeout)
    fields = [{**field, "type": DataType(field["type"]).name} for field in schema["fields"]]
    print_json(
        {
            "collection": a.collection,
            "fields": fields,
            "description": schema.get("description"),
            "properties": schema.get("properties"),
            "functions": schema.get("functions", []),
            "indexes": [
                c.describe_index(a.collection, name, timeout=a.timeout)
                for name in c.list_indexes(a.collection, timeout=a.timeout)
            ],
            "load_state": str(c.get_load_state(a.collection, timeout=a.timeout).get("state")),
            "stats": c.get_collection_stats(a.collection, timeout=a.timeout),
        }
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("health", "collections", "describe"):
        p = sub.add_parser(name)
        add_milvus_args(p)
        if name == "describe":
            p.add_argument("collection")
        p.set_defaults(handler=globals()["cmd_" + name])
    return parser


def main():
    a = build_parser().parse_args()
    c = connect(a)
    try:
        a.handler(c, a)
    finally:
        c.close()


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
