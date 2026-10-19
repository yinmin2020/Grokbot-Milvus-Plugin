# /// script
# requires-python = ">=3.10"
# dependencies = ["pymilvus>=2.6,<4", "pypdf>=4", "python-docx>=1.1", "fpdf2>=2.7"]
# ///
"""End-to-end test for the knowledge-base + ops scripts (milvus_kb.py, milvus_ops.py) against a real Milvus + embedding endpoint.

  MILVUS_URI=http://localhost:19530 EMBED_MODEL=embeddinggemma-2:740m uv run scripts/e2e_test.py
  [--metrics-url http://localhost:9091/metrics] [--keep]

Safety:
  * creates ONLY grokbot_kb_test_<ts> and grokbot_ops_test_<ts>, refuses if they already exist;
  * every collection that existed before the run is added to OPS_PROTECTED_COLLECTIONS, and writes are
    limited with OPS_WRITE_ALLOW_PREFIX=grokbot_kb_test_,grokbot_ops_test_;
  * at the end it drops only the two test collections (unless --keep) and verifies that every
    pre-existing collection still exists with the same row count.
"""
import argparse, json, os, random, shutil, subprocess, sys, tempfile, time, traceback

os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "false")  # we fork subprocesses while a gRPC channel is open
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from mvbot_common import connect, env

ap = argparse.ArgumentParser()
ap.add_argument("--uri", default=env("MILVUS_URI", "http://localhost:19530"))
ap.add_argument("--token", default=env("MILVUS_TOKEN", ""))
ap.add_argument("--db", default=env("MILVUS_DB", "default"))
ap.add_argument("--metrics-url", default=env("MILVUS_METRICS_URL"))
ap.add_argument("--keep", action="store_true", help="do not drop the test collections")
a = ap.parse_args()

ts = time.strftime("%m%d%H%M%S")
KB, OPS = f"grokbot_kb_test_{ts}", f"grokbot_ops_test_{ts}"
c = connect(a)


def count(n):  # read-only; never loads a collection
    if "Loaded" in str(c.get_load_state(n).get("state")):
        return ("count(*)", c.query(n, filter="", output_fields=["count(*)"])[0]["count(*)"])
    return ("stats", c.get_collection_stats(n).get("row_count"))


before = {n: count(n) for n in c.list_collections()}
for n in (KB, OPS):
    if n in before:
        sys.exit(f"REFUSED: {n} already exists")
ENV = {**os.environ, "MILVUS_URI": a.uri, "MILVUS_DB": a.db,
       "OPS_PROTECTED_COLLECTIONS": ",".join(sorted(before)),
       "OPS_WRITE_ALLOW_PREFIX": "grokbot_kb_test_,grokbot_ops_test_",
       "OPS_AUDIT_LOG": os.path.join(tempfile.gettempdir(), f"milvus-ops-audit-e2e-{ts}.log")}
if a.token:
    ENV["MILVUS_TOKEN"] = a.token
if a.metrics_url:
    ENV["MILVUS_METRICS_URL"] = a.metrics_url
results = []


def run(script, *args, cwd=None):
    p = subprocess.run([sys.executable, os.path.join(HERE, script), *args], capture_output=True, text=True,
                       env=ENV, cwd=cwd or ROOT, timeout=600)
    run.stdout = p.stdout.strip()
    return p.returncode, (p.stdout + p.stderr).strip()


def step(name, ok, out=""):
    results.append((name, bool(ok)))
    first = " | ".join(l for l in out.splitlines() if l.strip())[:220]
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"\n        {first}" if first else ""))


def kb(*args, cwd=None):
    return run("milvus_kb.py", *args, "--collection", KB, cwd=cwd)


def ops(*args):
    return run("milvus_ops.py", *args)


print(f"uri={a.uri} kb={KB} ops={OPS} protected(pre-existing)={sorted(before)}")
work = tempfile.mkdtemp(prefix="kb_e2e_")
try:
    # ---------------------------------------------------------------- Knowledge Desk
    docs = os.path.join(work, "docs")
    shutil.copytree(os.path.join(ROOT, "samples", "kb"), docs)
    subprocess.run([sys.executable, os.path.join(HERE, "make_sample_docs.py"), docs], check=True, capture_output=True)
    rc, out = kb("ingest", "docs", "--dry-run", cwd=work)
    step("kb ingest --dry-run (no collection created)", rc == 0 and not c.has_collection(KB) and "chunks=" in out, out)
    rc, out = kb("ingest", "docs", cwd=work)
    step("kb ingest 4 docs (md/txt/docx/pdf)", rc == 0 and '"added": 4' in out, out)
    rc, out = kb("ingest", "docs", cwd=work)
    step("kb re-ingest unchanged -> skipped", rc == 0 and '"unchanged": 4' in out, out)
    md = os.path.join(docs, "milvus_intro_zh.md")
    text = open(md, encoding="utf-8").read()
    open(md, "w", encoding="utf-8").write(text.split("## 度量类型")[0] + "\n## 新增章节\n\nGrokBot 测试标记：Milvus 的吉祥物是一只蓝色的鸟（测试数据）。\n")
    rc, out = kb("ingest", "docs", cwd=work)
    step("kb re-ingest changed doc -> upsert + stale chunks deleted", rc == 0 and '"updated": 1' in out and "stale" in out, out)
    rc, out = kb("info")
    info = json.loads(run.stdout) if rc == 0 else {}
    step("kb info (model+dim recorded in description)", rc == 0 and info.get("kb_meta", {}).get("dim") and info.get("documents") == 4,
         f"model={info.get('kb_meta', {}).get('embed_model')} dim={info.get('kb_meta', {}).get('dim')} rows={info.get('row_count')} docs={info.get('documents')}")
    rc, out = kb("list-docs")
    step("kb list-docs", rc == 0 and "total documents: 4" in out, out.splitlines()[-1] if out else "")
    cases = [("HNSW 索引的 M 和 efConstruction 一般设多少？", "milvus_intro_zh.md", True),
             ("What is the default k of RRFRanker?", "hybrid_search_en.txt", True),
             ("RRF 的默认 k 值是多少？", "hybrid_search_en.txt", True),
             ("备份之前需要先做什么？", "ops_manual_zh.docx", True),
             ("How many partitions can a collection have by default?", "partitions_memory_en.pdf", True),
             ("Milvus 的吉祥物是什么？", "milvus_intro_zh.md", True),
             ("如何做红烧肉？", None, False),
             ("What is the capital of France?", None, False)]
    for q, src, expect_found in cases:
        rc, out = kb("query", q, "--json", "--top-k", "3")
        if rc != 0:
            step(f"kb query: {q}", False, out); continue
        d = json.loads(run.stdout)
        srcs = [r["source"].rsplit("/", 1)[-1] for r in d["results"]]
        top = d["results"][0] if d["results"] else {}
        cite = f"best={d['best_dense_cosine']} found={d['found']} top=[1] {top.get('source')} > {top.get('heading') or ''} (p.{top.get('page')})"
        ok = (d["found"] and src in srcs) if expect_found else (not d["found"])
        step(f"kb query{' (expect NOT found)' if not expect_found else ''}: {q}", ok, cite)
    rc, out = kb("query", "BM25 analyzer jieba", "--mode", "bm25", "--top-k", "2")
    step("kb query --mode bm25", rc == 0 and "hybrid_search_en.txt" in out, out.splitlines()[0] if out else "")
    rc, out = kb("query", "partition key multi-tenant", "--ranker", "weighted", "--top-k", "2", "--filter", 'lang == "en"')
    step("kb query weighted ranker + filter lang==en", rc == 0 and "partitions_memory_en.pdf" in out, out.splitlines()[0] if out else "")
    rc, out = kb("delete-doc", "--source", "docs/hybrid_search_en.txt")
    step("kb delete-doc dry-run", rc == 0 and "[dry-run] would delete" in out, out)
    rc, out = kb("delete-doc", "--source", "docs/hybrid_search_en.txt", "--execute", "--confirm", "wrong")
    step("kb delete-doc wrong confirm -> refused", rc != 0 and "REFUSED" in out, out)
    rc, out = kb("delete-doc", "--source", "docs/hybrid_search_en.txt", "--execute", "--confirm", KB)
    rc2, out2 = kb("list-docs")
    step("kb delete-doc --execute --confirm", rc == 0 and "total documents: 3" in out2, out)

    # ---------------------------------------------------------------- Ops: read-only
    for args, must in [(["health"], '"version"'), (["collections"], "total:"), (["describe", KB], '"fields"'),
                       (["partitions", KB], "_default"), (["segments", KB], "persistent_count"),
                       (["metrics"], '"families"'), (["users"], '"roles"'), (["backup-guide"], "milvus-backup")]:
        rc, out = ops(*args)
        step("ops " + " ".join(args).replace(KB, "<kb>"), rc == 0 and must in out, out)

    # ---------------------------------------------------------------- Ops: guarded writes on own collection
    from pymilvus import DataType
    s = c.create_schema(auto_id=False)
    s.add_field("id", DataType.INT64, is_primary=True)
    s.add_field("tag", DataType.VARCHAR, max_length=32)
    s.add_field("vec", DataType.FLOAT_VECTOR, dim=8)
    ip = c.prepare_index_params(); ip.add_index("vec", index_type="AUTOINDEX", metric_type="COSINE")
    c.create_collection(OPS, schema=s, index_params=ip)
    for batch in range(3):
        c.insert(OPS, [{"id": batch * 100 + i, "tag": random.choice(["a", "b"]), "vec": [random.random() for _ in range(8)]} for i in range(100)])
    step("ops test collection created (300 rows)", c.has_collection(OPS))
    rc, out = ops("flush", OPS); step("ops flush dry-run", rc == 0 and "[dry-run]" in out, out)
    rc, out = ops("flush", OPS, "--execute"); step("ops flush --execute", rc == 0 and "[done]" in out, out)
    rc, out = ops("index-create", OPS, "--field", "tag", "--index-type", "INVERTED")
    step("ops index-create dry-run", rc == 0 and "[dry-run]" in out, out)
    rc, out = ops("index-create", OPS, "--field", "tag", "--index-type", "INVERTED", "--execute")
    step("ops index-create --execute", rc == 0 and "[done]" in out, out)
    rc, out = ops("partition-create", OPS, "p2026", "--execute"); step("ops partition-create --execute", rc == 0 and "[done]" in out, out)
    rc, out = ops("load", OPS, "--execute"); step("ops load --execute", rc == 0 and "Loaded" in out, out)
    rc, out = ops("delete-entities", OPS, "--filter", 'tag == "a"')
    step("ops delete-entities dry-run (shows match count)", rc == 0 and "[dry-run]" in out and "matching now:" in out, out)
    rc, out = ops("delete-entities", OPS, "--filter", 'tag == "a"', "--execute")
    step("ops delete-entities --execute without --confirm -> refused", rc != 0 and "REFUSED" in out, out)
    rc, out = ops("delete-entities", OPS, "--filter", 'tag == "a"', "--execute", "--confirm", OPS)
    step("ops delete-entities --execute --confirm", rc == 0 and "[done]" in out, out)
    rc, out = ops("compact", OPS, "--execute", "--wait", "--wait-timeout", "90")
    step("ops compact --execute --wait", rc == 0 and "job_id" in out, out)
    try:
        so = run.stdout
        job = json.loads(so[so.index("{"):])["job_id"]
        rc, out = ops("compaction-state", str(job)); step("ops compaction-state", rc == 0 and '"state"' in out, out)
    except Exception as e:
        step("ops compaction-state", False, str(e))
    rc, out = ops("segments", OPS); step("ops segments (own collection)", rc == 0 and "persistent_count" in out, out)
    rc, out = ops("describe", OPS); step("ops describe (own collection)", rc == 0 and "p2026" in out and "INVERTED" in out, out)
    rc, out = ops("release", OPS); step("ops release dry-run", rc == 0 and "[dry-run]" in out, out)
    rc, out = ops("release", OPS, "--execute", "--confirm", "nope"); step("ops release wrong confirm -> refused", rc != 0 and "REFUSED" in out, out)
    rc, out = ops("release", OPS, "--execute", "--confirm", OPS); step("ops release --execute --confirm", rc == 0 and "NotLoad" in out, out)
    idx = [i for i in c.list_indexes(OPS) if c.describe_index(OPS, i).get("field_name") == "tag"][0]
    rc, out = ops("index-drop", OPS, "--index", idx); step("ops index-drop dry-run", rc == 0 and "[dry-run]" in out, out)
    rc, out = ops("index-drop", OPS, "--index", idx, "--execute", "--confirm", OPS)
    step("ops index-drop --execute --confirm", rc == 0 and "[done]" in out and idx not in c.list_indexes(OPS), out)
    rc, out = ops("partition-drop", OPS, "p2026", "--execute", "--confirm", OPS)
    step("ops partition-drop --execute --confirm", rc == 0 and not c.has_partition(OPS, "p2026"), out)
    # guards against other collections (all refuse before touching anything)
    if before:
        victim = sorted(before)[0]
        rc, out = ops("drop-collection", victim, "--execute", "--confirm", victim)
        step(f"guard: drop-collection on protected '{victim}' -> refused", rc != 0 and "protected" in out, out)
        rc, out = ops("release", victim, "--execute", "--confirm", victim)
        step(f"guard: release on protected '{victim}' -> refused", rc != 0 and "REFUSED" in out, out)
    rc, out = run("milvus_ops.py", "flush", "some_other_collection", "--execute")
    step("guard: OPS_WRITE_ALLOW_PREFIX blocks writes outside test prefixes", rc != 0 and "REFUSED" in out, out)
except Exception:
    step("unexpected exception", False, traceback.format_exc()[-600:].replace("\n", " | "))
finally:
    shutil.rmtree(work, ignore_errors=True)
    if not a.keep:
        for n in (KB, OPS):
            if c.has_collection(n):
                rc, out = ops("drop-collection", n)
                step(f"cleanup: drop-collection {n} dry-run", rc == 0 and "[dry-run]" in out, out)
                rc, out = ops("drop-collection", n, "--execute", "--confirm", n)
                step(f"cleanup: drop-collection {n} --execute --confirm", rc == 0 and not c.has_collection(n), out)
    after = set(c.list_collections())
    missing = [n for n in before if n not in after]
    changed = []
    for n in before:
        if n in after and count(n) != before[n]:
            changed.append((n, before[n], count(n)))
    step("pre-existing collections untouched (exist + same row count)", not missing and not changed,
         f"before={before} missing={missing} changed={changed}")
    print(f"\nRESULT: {sum(ok for _, ok in results)}/{len(results)} passed; audit log: {ENV['OPS_AUDIT_LOG']}")
    sys.exit(0 if all(ok for _, ok in results) else 1)
