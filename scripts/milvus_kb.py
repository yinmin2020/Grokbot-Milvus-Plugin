# /// script
# requires-python = ">=3.10"
# dependencies = ["pymilvus>=2.6,<4", "pypdf>=4", "python-docx>=1.1"]
# ///
"""Knowledge-base (Knowledge Desk / 知识库管家) toolkit: ingest documents into a Milvus hybrid
(dense + BM25) collection and answer questions with cited chunks.

  uv run scripts/milvus_kb.py init      --collection kb_docs --embed-model embeddinggemma-2:740m
  uv run scripts/milvus_kb.py ingest    docs/ handbook.pdf --collection kb_docs
  uv run scripts/milvus_kb.py query     "如何创建 HNSW 索引？" --collection kb_docs
  uv run scripts/milvus_kb.py info      --collection kb_docs
  uv run scripts/milvus_kb.py list-docs --collection kb_docs
  uv run scripts/milvus_kb.py delete-doc --source docs/old.md --collection kb_docs          # dry-run
  uv run scripts/milvus_kb.py delete-doc --source docs/old.md --collection kb_docs --execute --confirm kb_docs

Supported files: .md .markdown .txt .pdf .docx. Re-ingesting is idempotent: a document whose
content hash is unchanged is skipped; a changed document is upserted (deterministic pks) and its
stale chunks are deleted. The embedding provider/model/dim/prefixes are recorded in the collection
description, and query/ingest refuse to mix models.
Connection / embedding settings: see mvbot_common.py (MILVUS_URI, EMBED_* env vars).
"""
import argparse, hashlib, json, os, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mvbot_common import (Embedder, add_embed_args, add_milvus_args, connect,
                          default_prefixes, env, print_json, redact_uri)

SCHEMA_TAG = "mvbot-kb/v1"
CHUNKER_VERSION = "1"
DEFAULT_CHUNK, DEFAULT_OVERLAP = int(os.environ.get("KB_CHUNK_SIZE", "500")), int(os.environ.get("KB_CHUNK_OVERLAP", "80"))
EXTS = {".md", ".markdown", ".txt", ".pdf", ".docx"}
ANALYZER = {"tokenizer": "jieba", "filter": ["lowercase", "cnalphanumonly"]}  # zh + en
OUT_FIELDS = ["pk", "doc_id", "source", "title", "heading", "page", "chunk_no", "lang", "text"]
CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")


# ------------------------------------------------------------------ parsing
def _paras(text):
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def parse_markdown(text, fallback_title):
    blocks, stack, buf, in_code, title = [], [], [], False, None

    def flush():
        if buf:
            body = "\n".join(buf).strip()
            if body:
                blocks.append((" > ".join(h for _, h in stack), None, body))
            buf.clear()

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            buf.append(line)
            continue
        m = None if in_code else re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if m:
            flush()
            level, h = len(m.group(1)), m.group(2).strip()
            if level == 1 and title is None:
                title = h
            stack[:] = [(l, x) for l, x in stack if l < level] + [(level, h)]
        elif not line.strip() and not in_code:
            flush()
        else:
            buf.append(line)
    flush()
    return title or fallback_title, blocks


def parse_txt(text, fallback_title):
    ps = _paras(text)
    if len(ps) > 1 and "\n" not in ps[0] and len(ps[0]) <= 120:  # short first line = title
        return ps[0], [("", None, p) for p in ps[1:]]
    return fallback_title, [("", None, p) for p in ps]


def parse_pdf(path, fallback_title):
    from pypdf import PdfReader
    r = PdfReader(path)
    title = (r.metadata.title if r.metadata and r.metadata.title else None) or fallback_title
    blocks = []
    for i, page in enumerate(r.pages, 1):
        t = page.extract_text() or ""
        ps = _paras(t) or [l.strip() for l in t.splitlines() if l.strip()]
        blocks += [("", i, re.sub(r"[ \t]*\n[ \t]*", "\n", p)) for p in ps]
    return title, blocks


def parse_docx(path, fallback_title):
    import docx
    d = docx.Document(path)
    title = (d.core_properties.title or "").strip() or None
    blocks, stack = [], []
    for p in d.paragraphs:
        t = p.text.strip()
        if not t:
            continue
        style = (p.style.name if p.style is not None else "") or ""
        m = re.match(r"^(Heading|标题)\s*(\d)", style)
        if style == "Title":
            title = title or t
            continue
        if m:
            level = int(m.group(2))
            if level == 1 and title is None:
                title = t
            stack[:] = [(l, x) for l, x in stack if l < level] + [(level, t)]
            continue
        blocks.append((" > ".join(h for _, h in stack), None, t))
    for ti, table in enumerate(d.tables, 1):
        rows = [" | ".join(c.text.strip() for c in row.cells) for row in table.rows]
        if rows:
            blocks.append((f"table {ti}", None, "\n".join(rows)))
    return title or fallback_title, blocks


def parse_file(path):
    ext = os.path.splitext(path)[1].lower()
    fallback = os.path.splitext(os.path.basename(path))[0]
    if ext == ".pdf":
        return parse_pdf(path, fallback)
    if ext == ".docx":
        return parse_docx(path, fallback)
    text = open(path, encoding="utf-8", errors="replace").read()
    if ext in (".md", ".markdown"):
        return parse_markdown(text, fallback)
    return parse_txt(text, fallback)


# ------------------------------------------------------------------ chunking
_SENT = re.compile(r"(?<=[。！？!?；;])|(?<=[.!?])\s+")


def _split_long(text, size):
    sents = [s for s in _SENT.split(text) if s and s.strip()]
    out, cur = [], ""
    for s in sents:
        while len(s) > size:  # hard split very long sentences
            if cur:
                out.append(cur); cur = ""
            out.append(s[:size]); s = s[size:]
        if len(cur) + len(s) > size and cur:
            out.append(cur); cur = ""
        sep = "" if (not cur or cur.endswith((" ", "\n")) or CJK.search(cur[-1:]) or CJK.search(s[:1])) else " "
        cur += sep + s
    if cur.strip():
        out.append(cur)
    return out


def _tail(text, overlap):
    if overlap <= 0 or len(text) <= overlap:
        return text if overlap > 0 else ""
    t = text[-overlap:]
    m = re.search(r"[。！？!?；;.]\s*", t)  # start the overlap at a sentence boundary when possible
    return t[m.end():] if m and m.end() < len(t) - 10 else t


def chunk_blocks(blocks, size, overlap):
    """blocks: [(heading, page, text)] -> [(heading, page, chunk_text)]; never crosses a section."""
    chunks, i = [], 0
    while i < len(blocks):
        heading, page = blocks[i][0], blocks[i][1]
        sec = []
        while i < len(blocks) and blocks[i][0] == heading and blocks[i][1] == page:
            sec.append(blocks[i][2]); i += 1
        pieces = []
        for p in sec:
            lim = max(50, size - overlap)  # leave room for the overlap prefix
            pieces += _split_long(p, lim) if len(p) > lim else [p]
        cur = ""
        for p in pieces:
            if cur and len(cur) + len(p) + 1 > size:
                chunks.append((heading, page, cur))
                tail = _tail(cur, overlap)
                cur = (tail + "\n" if tail else "") + p
            else:
                cur = (cur + "\n" + p) if cur else p
        if cur.strip():
            chunks.append((heading, page, cur))
    return chunks


def lang_of(text):
    if not text:
        return "und"
    return "zh" if len(CJK.findall(text)) / max(1, len(text)) > 0.15 else "en"


def clip_bytes(s, n):
    b = s.encode("utf-8")
    return s if len(b) <= n else b[:n].decode("utf-8", errors="ignore")


# ------------------------------------------------------------------ collection
def read_meta(c, col):
    d = c.describe_collection(col)
    try:
        meta = json.loads(d.get("description") or "{}")
    except json.JSONDecodeError:
        meta = {}
    return d, meta if meta.get("schema") == SCHEMA_TAG else None


def create_collection(c, col, emb, dim, prefixes, a):
    from pymilvus import DataType, Function, FunctionType
    meta = {"schema": SCHEMA_TAG, "embed_provider": emb.provider, "embed_model": emb.model, "dim": dim,
            "metric": "COSINE", "query_prefix": prefixes["query"], "doc_prefix": prefixes["doc"],
            "min_score": prefixes.get("min_score", 0.5),
            "chunk_size": a.chunk_size or DEFAULT_CHUNK, "chunk_overlap": a.chunk_overlap if a.chunk_overlap is not None else DEFAULT_OVERLAP, "analyzer": ANALYZER,
            "created_at": int(time.time())}
    s = c.create_schema(auto_id=False, enable_dynamic_field=False, description=json.dumps(meta, ensure_ascii=False))
    s.add_field("pk", DataType.VARCHAR, is_primary=True, max_length=160)
    s.add_field("doc_id", DataType.VARCHAR, max_length=40)
    s.add_field("doc_hash", DataType.VARCHAR, max_length=64)
    s.add_field("source", DataType.VARCHAR, max_length=1024)
    s.add_field("title", DataType.VARCHAR, max_length=512)
    s.add_field("heading", DataType.VARCHAR, max_length=1024)
    s.add_field("page", DataType.INT64)
    s.add_field("chunk_no", DataType.INT64)
    s.add_field("lang", DataType.VARCHAR, max_length=8)
    s.add_field("updated_at", DataType.INT64)
    s.add_field("text", DataType.VARCHAR, max_length=65535, enable_analyzer=True, enable_match=True,
                analyzer_params=ANALYZER)
    s.add_field("dense", DataType.FLOAT_VECTOR, dim=dim)
    s.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    s.add_function(Function(name="text_bm25", function_type=FunctionType.BM25,
                            input_field_names=["text"], output_field_names=["sparse"]))
    ip = c.prepare_index_params()
    ip.add_index("dense", index_type="AUTOINDEX", metric_type="COSINE")
    ip.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25",
                 params={"inverted_index_algo": "DAAT_MAXSCORE"})
    ip.add_index("doc_id", index_type="INVERTED")
    ip.add_index("source", index_type="INVERTED")
    c.create_collection(col, schema=s, index_params=ip)
    return meta


def resolve_embedder(c, a, create=False):
    """Return (embedder, meta). Uses the model recorded in the collection unless overridden;
    refuses to mix models. If create=True and the collection is missing, creates it."""
    col = a.collection
    exists = c.has_collection(col)
    meta = None
    if exists:
        _, meta = read_meta(c, col)
        if meta is None:
            sys.exit(f"ERROR: collection '{col}' exists but was not created by this tool (no {SCHEMA_TAG} "
                     "description). Use another --collection name.")
        provider = a.embed_provider or meta["embed_provider"]
        model = a.embed_model or meta["embed_model"]
        if model == meta["embed_model"] and provider != meta["embed_provider"]:
            print(f"WARNING: collection was built via provider '{meta['embed_provider']}', now using '{provider}' "
                  f"with the same model '{model}'. OK only if both serve the same model (e.g. Ollama /api/embed vs /v1).",
                  file=sys.stderr)
        elif model != meta["embed_model"]:
            sys.exit(f"ERROR: collection '{col}' was built with {meta['embed_provider']}:{meta['embed_model']} "
                     f"(dim {meta['dim']}); refusing to use {provider}:{model}. Re-ingest into a new collection.")
        emb = Embedder(provider, a.embed_endpoint, model, a.embed_batch, a.embed_timeout)
        return emb, meta
    if not create:
        sys.exit(f"ERROR: collection '{col}' does not exist. Run `init` or `ingest` first.")
    emb = Embedder(a.embed_provider, a.embed_endpoint, a.embed_model, a.embed_batch, a.embed_timeout)
    dim = emb.dim()
    prefixes = default_prefixes(emb.model)
    if a.query_prefix is not None:
        prefixes["query"] = a.query_prefix
    if a.doc_prefix is not None:
        prefixes["doc"] = a.doc_prefix
    meta = create_collection(c, col, emb, dim, prefixes, a)
    print(f"[created] collection '{col}' dense dim={dim} model={emb.provider}:{emb.model} + BM25 sparse")
    return emb, meta


def ensure_loaded(c, col):
    st = str(c.get_load_state(col).get("state"))
    if "Loaded" not in st:
        c.load_collection(col)


# ------------------------------------------------------------------ commands
def iter_files(paths):
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                dirs[:] = sorted(d for d in dirs if not d.startswith("."))
                for f in sorted(files):
                    if os.path.splitext(f)[1].lower() in EXTS and not f.startswith((".", "~$")):
                        yield os.path.join(root, f)
        elif os.path.isfile(p):
            if os.path.splitext(p)[1].lower() in EXTS:
                yield p
            else:
                print(f"[skip] unsupported file type: {p}")
        else:
            print(f"[skip] not found: {p}")


def source_name(path, prefix):
    ap = os.path.abspath(path)
    rel = os.path.relpath(ap, os.getcwd())
    name = ap if rel.startswith("..") else rel
    name = name.replace(os.sep, "/")
    return (prefix.rstrip("/") + "/" + name) if prefix else name


def esc(s):
    return s.replace("\\", "\\\\").replace('"', '\\"')


def cmd_init(c, a):
    if c.has_collection(a.collection):
        _, meta = read_meta(c, a.collection)
        print(f"[exists] '{a.collection}' meta={json.dumps(meta, ensure_ascii=False)}")
        return
    resolve_embedder(c, a, create=True)


def cmd_ingest(c, a):
    if a.dry_run and not c.has_collection(a.collection):
        emb, meta = None, {"chunk_size": DEFAULT_CHUNK, "chunk_overlap": DEFAULT_OVERLAP, "doc_prefix": ""}
        print(f"[dry-run] collection '{a.collection}' does not exist yet; it would be created on a real run")
    else:
        emb, meta = resolve_embedder(c, a, create=True)
    # explicit CLI values win; otherwise use what the collection was created with
    size = a.chunk_size or meta["chunk_size"]
    overlap = a.chunk_overlap if a.chunk_overlap is not None else meta["chunk_overlap"]
    if overlap >= size or size < 100:
        sys.exit("ERROR: need chunk_size >= 100 and chunk_overlap < chunk_size")
    files = list(iter_files(a.paths))
    if not files:
        sys.exit("ERROR: no supported files found")
    summary = {"added": 0, "updated": 0, "unchanged": 0, "empty": 0, "chunks_written": 0, "stale_deleted": 0}
    for path in files:
        src = source_name(path, a.source_prefix)
        doc_id = hashlib.sha1(src.encode()).hexdigest()[:20]
        raw = open(path, "rb").read()
        doc_hash = hashlib.sha256(raw + f"|{size}|{overlap}|{CHUNKER_VERSION}".encode()).hexdigest()
        old = c.query(a.collection, filter=f'doc_id == "{doc_id}"', output_fields=["doc_hash"], limit=1) if emb else []
        if old and old[0]["doc_hash"] == doc_hash and not a.force:
            summary["unchanged"] += 1
            print(f"[unchanged] {src}")
            continue
        title, blocks = parse_file(path)
        chunks = chunk_blocks(blocks, size, overlap)
        if not chunks:
            summary["empty"] += 1
            print(f"[empty] {src} (no extractable text; scanned PDF? OCR it first)")
            continue
        now = int(time.time())
        rows, inputs = [], []
        for n, (heading, page, body) in enumerate(chunks):
            text = (f"[{heading}]\n" if heading else "") + body
            rows.append({"pk": f"{doc_id}#{n:05d}", "doc_id": doc_id, "doc_hash": doc_hash,
                         "source": clip_bytes(src, 1024), "title": clip_bytes(title, 512),
                         "heading": clip_bytes(heading, 1024), "page": page or 0, "chunk_no": n,
                         "lang": lang_of(body), "updated_at": now, "text": clip_bytes(text, 65000)})
            inputs.append(meta["doc_prefix"].replace("{title}", title or "none") + text)
        if a.dry_run:
            print(f"[dry-run] {src}: title={title!r} chunks={len(rows)} (not written)")
            continue
        vecs = emb.embed(inputs)
        if len(vecs[0]) != meta["dim"]:
            sys.exit(f"ERROR: model returned dim {len(vecs[0])} but collection expects {meta['dim']}")
        for r, v in zip(rows, vecs):
            r["dense"] = v
        for i in range(0, len(rows), 64):
            c.upsert(a.collection, rows[i:i + 64])
        stale = c.query(a.collection, filter=f'doc_id == "{doc_id}" and doc_hash != "{doc_hash}"',
                        output_fields=["pk"], limit=16384)
        if stale:
            c.delete(a.collection, filter=f'doc_id == "{doc_id}" and doc_hash != "{doc_hash}"')
        kind = "updated" if old else "added"
        summary[kind] += 1
        summary["chunks_written"] += len(rows)
        summary["stale_deleted"] += len(stale)
        print(f"[{kind}] {src}: {len(rows)} chunks" + (f", {len(stale)} stale chunks deleted" if stale else ""))
    print("summary:", json.dumps(summary))


def _hit(h):
    """pymilvus 2.x/3.x compatible (pk, distance, entity) for a search hit."""
    e = h.get("entity") if isinstance(h, dict) and isinstance(h.get("entity"), dict) else None
    e = e or (h.entity if hasattr(h, "entity") and isinstance(getattr(h, "entity"), dict) else {}) or h
    pk = getattr(h, "id", None) or getattr(h, "pk", None)
    if pk is None:
        for k in ("id", "pk"):
            try:
                pk = h[k]; break
            except (KeyError, TypeError):
                pass
    if pk is None:
        pk = e.get("pk")
    dist = getattr(h, "distance", None)
    if dist is None:
        dist = h["distance"]
    return pk, float(dist), e


def cmd_query(c, a):
    from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker
    emb, meta = resolve_embedder(c, a)
    if a.min_score is None:
        a.min_score = float(meta.get("min_score", 0.5))
    ensure_loaded(c, a.collection)
    cand = max(a.candidates, a.top_k)
    flt = a.filter or ""
    qvec = emb.embed([meta["query_prefix"] + a.question])[0]
    dense = c.search(a.collection, data=[qvec], anns_field="dense", limit=cand, filter=flt,
                     search_params={"metric_type": "COSINE"}, output_fields=OUT_FIELDS)[0]
    dense_score = {_hit(h)[0]: _hit(h)[1] for h in dense}
    if a.mode == "dense":
        hits = list(dense)[:a.top_k]
    elif a.mode == "bm25":
        hits = c.search(a.collection, data=[a.question], anns_field="sparse", limit=a.top_k, filter=flt,
                        search_params={"metric_type": "BM25"}, output_fields=OUT_FIELDS)[0]
    else:
        reqs = [AnnSearchRequest(data=[qvec], anns_field="dense", param={"metric_type": "COSINE"}, limit=cand, expr=flt),
                AnnSearchRequest(data=[a.question], anns_field="sparse", param={"metric_type": "BM25"}, limit=cand, expr=flt)]
        ranker = RRFRanker(a.rrf_k) if a.ranker == "rrf" else WeightedRanker(*[float(x) for x in a.weights.split(",")])
        hits = c.hybrid_search(a.collection, reqs, ranker, limit=a.top_k, output_fields=OUT_FIELDS)[0]
    results = []
    for i, h in enumerate(hits, 1):
        pk, dist, e = _hit(h)
        results.append({"cite": i, "pk": pk, "score": round(dist, 4),
                        "dense_cosine": round(float(dense_score[pk]), 4) if pk in dense_score else None,
                        "source": e.get("source"), "title": e.get("title"), "heading": e.get("heading"),
                        "page": e.get("page") or None, "chunk_no": e.get("chunk_no"), "lang": e.get("lang"),
                        "text": e.get("text")})
    best = max([r["dense_cosine"] for r in results if r["dense_cosine"] is not None] or [0.0])
    found = bool(results) and best >= a.min_score
    out = {"question": a.question, "collection": a.collection, "mode": a.mode + (f"/{a.ranker}" if a.mode == "hybrid" else ""),
           "embed_model": f"{meta['embed_provider']}:{meta['embed_model']}", "best_dense_cosine": round(best, 4),
           "min_score": a.min_score, "found": found, "results": results}
    if a.json:
        print_json(out)
        return
    print(f"Q: {a.question}\nmode={a.mode} model={out['embed_model']} best_dense_cosine={out['best_dense_cosine']} "
          f"threshold={a.min_score} -> {'FOUND' if found else 'NOT_FOUND (answer: 知识库中没有找到相关内容)'}")
    for r in results:
        loc = r["source"] + (f" > {r['heading']}" if r["heading"] else "") + (f" (p.{r['page']})" if r["page"] else "")
        print(f"\n[{r['cite']}] {loc}\n    pk={r['pk']} score={r['score']} dense_cosine={r['dense_cosine']}")
        body = r["text"] if a.full else (r["text"][:300] + ("…" if len(r["text"]) > 300 else ""))
        print("    " + body.replace("\n", "\n    "))


def _docs(c, col, flt=""):
    docs = {}
    it = c.query_iterator(col, batch_size=1000, filter=flt,
                          output_fields=["doc_id", "source", "title", "doc_hash", "updated_at"])
    while True:
        batch = it.next()
        if not batch:
            it.close(); break
        for r in batch:
            d = docs.setdefault(r["doc_id"], {"doc_id": r["doc_id"], "source": r["source"], "title": r["title"],
                                              "doc_hash": r["doc_hash"][:12], "updated_at": r["updated_at"], "chunks": 0})
            d["chunks"] += 1
    return sorted(docs.values(), key=lambda d: d["source"])


def cmd_info(c, a):
    d, meta = read_meta(c, a.collection)
    ensure_loaded(c, a.collection)
    rows = c.query(a.collection, filter="", output_fields=["count(*)"])[0]["count(*)"]
    docs = _docs(c, a.collection)
    print_json({"collection": a.collection, "uri": redact_uri(a.uri), "kb_meta": meta,
                "row_count": rows, "documents": len(docs),
                "load_state": str(c.get_load_state(a.collection).get("state")),
                "fields": [f["name"] for f in d["fields"]], "functions": [f.get("name") for f in d.get("functions", [])]})


def cmd_list_docs(c, a):
    ensure_loaded(c, a.collection)
    docs = _docs(c, a.collection)
    for d in docs:
        print(f"{d['doc_id']}  chunks={d['chunks']:<4} hash={d['doc_hash']}  {d['source']}  ({d['title']})")
    print(f"total documents: {len(docs)}")


def cmd_delete_doc(c, a):
    if not (a.source or a.doc_id):
        sys.exit("ERROR: pass --source or --doc-id")
    flt = f'doc_id == "{esc(a.doc_id)}"' if a.doc_id else f'source == "{esc(a.source)}"'
    ensure_loaded(c, a.collection)
    n = len(c.query(a.collection, filter=flt, output_fields=["pk"], limit=16384))
    if not a.execute:
        print(f"[dry-run] would delete {n} chunks from '{a.collection}' where {flt}. "
              f"Re-run with --execute --confirm {a.collection}")
        return
    if a.confirm != a.collection:
        sys.exit(f"REFUSED: --confirm must equal the collection name '{a.collection}'")
    c.delete(a.collection, filter=flt)
    print(f"[deleted] {n} chunks where {flt}")


def main():
    default_col = env("KB_COLLECTION", "kb_docs")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, embed=True):
        add_milvus_args(p)
        p.add_argument("--collection", default=default_col)
        if embed:
            add_embed_args(p)
        return p

    for name in ("init", "ingest"):
        p = common(sub.add_parser(name))
        p.add_argument("--chunk-size", type=int, default=None, help="characters per chunk (default: collection setting, new collections 500 or env KB_CHUNK_SIZE)")
        p.add_argument("--chunk-overlap", type=int, default=None, help="default: collection setting, new collections 80 or env KB_CHUNK_OVERLAP")
        p.add_argument("--query-prefix", default=None, help="override model task prefix for queries")
        p.add_argument("--doc-prefix", default=None, help="override model task prefix for documents ({title} allowed)")
        if name == "ingest":
            p.add_argument("paths", nargs="+")
            p.add_argument("--source-prefix", default="", help="e.g. a URL base stored in `source`")
            p.add_argument("--force", action="store_true", help="re-embed even if unchanged")
            p.add_argument("--dry-run", action="store_true", help="parse + chunk only, write nothing")
    p = common(sub.add_parser("query"))
    p.add_argument("question")
    p.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default="hybrid")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--candidates", type=int, default=20)
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--filter", default="", help='Milvus filter, e.g. lang == "zh" or source like "docs/%%"')
    p.add_argument("--min-score", type=float, default=float(env("KB_MIN_SCORE")) if env("KB_MIN_SCORE") else None,
                   help="best dense cosine below this => found=false (default: collection min_score, else 0.5)")
    p.add_argument("--ranker", choices=["rrf", "weighted"], default="rrf")
    p.add_argument("--weights", default="0.7,0.3", help="dense,bm25 weights for --ranker weighted")
    p.add_argument("--json", action="store_true")
    p.add_argument("--full", action="store_true", help="print full chunk text")
    common(sub.add_parser("info"), embed=False)
    common(sub.add_parser("list-docs"), embed=False)
    p = common(sub.add_parser("delete-doc"), embed=False)
    p.add_argument("--source"); p.add_argument("--doc-id")
    p.add_argument("--execute", action="store_true"); p.add_argument("--confirm", default="")
    a = ap.parse_args()
    c = connect(a)
    if a.cmd not in ("init", "ingest") and not c.has_collection(a.collection):
        sys.exit(f"ERROR: collection '{a.collection}' does not exist")
    {"init": cmd_init, "ingest": cmd_ingest, "query": cmd_query, "info": cmd_info,
     "list-docs": cmd_list_docs, "delete-doc": cmd_delete_doc}[a.cmd](c, a)


if __name__ == "__main__":
    main()
