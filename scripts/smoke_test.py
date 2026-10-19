# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp<2", "pymilvus>=2.5", "milvus-lite; sys_platform != 'win32'"]
# ///
"""End-to-end smoke test for the Milvus plugin.

Launches the MCP server exactly as declared in ../mcp.json (stdio), lists its
tools, then creates a throwaway collection, inserts a few vectors, loads it,
runs a vector search + filtered query, and finally drops ONLY that collection.

Usage:
  uv run scripts/smoke_test.py --uri http://localhost:19530
  MILVUS_TOKEN=<api-key> uv run scripts/smoke_test.py --uri https://<endpoint>.zillizcloud.com
  uv run scripts/smoke_test.py --uri /absolute/path/demo.db        # Milvus Lite (macOS/Linux)
  uv run scripts/smoke_test.py --uri ... --list-only               # only start + list tools

Safety: refuses to run if the test collection already exists, and never
touches any other collection. The MCP server has no drop tool, so the final
cleanup uses pymilvus directly (drop_collection on the test collection only).
"""
import argparse, asyncio, json, os, re, sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--uri", required=True)
ap.add_argument("--token", default=os.environ.get("MILVUS_TOKEN", ""))
ap.add_argument("--db", default=os.environ.get("MILVUS_DB", "default"))
ap.add_argument("--collection", default="grokbot_plugin_test")
ap.add_argument("--list-only", action="store_true")
a = ap.parse_args()

cfg = json.load(open(os.path.join(HERE, "..", "mcp.json")))["mcpServers"]["milvus"]
vals = {"MILVUS_URI": a.uri, "MILVUS_TOKEN": a.token, "MILVUS_DB": a.db}
sub = lambda s: re.sub(r"\$\{(\w+)\}", lambda m: vals[m.group(1)], s)
args = [sub(x) for x in cfg["args"]]
env = {**os.environ, **{k: sub(v) for k, v in cfg.get("env", {}).items()}}
env.pop("MILVUS_URI", None)  # pymilvus validates this env var at import time
C = a.collection
created = False
fail = 0


async def main():
    global created, fail
    params = StdioServerParameters(command=cfg["command"], args=args, env=env)
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = [t.name for t in (await s.list_tools()).tools]
            print(f"[ok] server started, {len(tools)} tools: {', '.join(tools)}")
            if a.list_only:
                return

            async def call(name, expect=None, **kw):
                global fail
                res = await s.call_tool(name, kw)
                txt = res.content[0].text if res.content else ""
                ok = not txt.startswith("Error") and (expect is None or expect in txt)
                fail += 0 if ok else 1
                print(f"[{'ok' if ok else 'FAIL'}] {name}: {txt.strip()[:300]}")
                return txt

            cols = await call("milvus_list_collections")
            if re.search(rf"(^|[\s,:]){re.escape(C)}(,|$|\s)", cols):
                print(f"[abort] collection '{C}' already exists; not touching it.")
                fail += 1
                return
            await call("milvus_create_collection", expect="created", collection_name=C, dimension=4)
            created = True
            await call("milvus_insert_data", expect="insert_count", collection_name=C, data=[
                {"vector": [0.1, 0.2, 0.3, 0.4], "text": "Milvus is an open-source vector database", "tag": "db"},
                {"vector": [0.9, 0.1, 0.0, 0.2], "text": "Cats like to sleep", "tag": "pets"},
                {"vector": [0.2, 0.2, 0.3, 0.5], "text": "Zilliz Cloud is managed Milvus", "tag": "db"}])
            await call("milvus_load_collection", collection_name=C)
            await call("milvus_vector_search", expect="Milvus is an open-source", collection_name=C,
                       vector=[0.1, 0.2, 0.3, 0.4], limit=2, output_fields=["text", "tag"])
            await call("milvus_vector_search", expect="Cats", collection_name=C, vector=[0.1, 0.2, 0.3, 0.4],
                       limit=3, output_fields=["text"], filter_expr='tag == "pets"')
            await call("milvus_query", expect="Zilliz", collection_name=C, filter_expr='tag == "db"', output_fields=["text"])
            await call("milvus_get_collection_info", expect=C, collection_name=C)


try:
    asyncio.run(main())
finally:
    if created:
        from pymilvus import MilvusClient
        cl = MilvusClient(uri=a.uri, token=a.token, db_name=a.db)
        cl.drop_collection(C)
        print(f"[ok] dropped test collection '{C}' (only).")
        cl.close()
print("RESULT:", "PASS" if fail == 0 else f"{fail} FAILURE(S)")
sys.exit(1 if fail else 0)
