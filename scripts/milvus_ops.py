# /// script
# requires-python = ">=3.10"
# dependencies = ["pymilvus>=2.6,<4"]
# ///
"""Milvus ops toolkit (运维助手): safe, scriptable Milvus operations.

READ (always safe):
  health | collections | describe COL | partitions COL | segments COL | compaction-state JOB_ID
  metrics [--grep REGEX] | users | backup-guide
WRITE (dry-run unless --execute):
  index-create COL --field F [--index-type T --metric M --params JSON] | load COL [--partitions P..]
  partition-create COL P | flush COL | compact COL [--wait]
DESTRUCTIVE (dry-run unless --execute AND --confirm <collection name>):
  release COL | index-drop COL --index NAME | partition-drop COL P
  delete-entities COL --filter EXPR | drop-collection COL

Guards:
  * protected collections (env OPS_PROTECTED_COLLECTIONS, comma separated) can never be changed by
    WRITE/DESTRUCTIVE commands;
  * env OPS_WRITE_ALLOW_PREFIX (comma separated) restricts every write to collections with those prefixes;
  * every executed write is appended to OPS_AUDIT_LOG (default ~/.milvus-ops-audit.log), no secrets.
Connection: MILVUS_URI / MILVUS_TOKEN / MILVUS_DB; metrics: MILVUS_METRICS_URL (default http://<host>:9091/metrics).
"""
import argparse, json, os, re, sys, time, urllib.parse, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mvbot_common import add_milvus_args, connect, env, print_json, redact_uri, urlopen

WRITE = {"index-create", "load", "partition-create", "flush", "compact"}
DESTRUCTIVE = {"release", "index-drop", "partition-drop", "delete-entities", "drop-collection"}


# ------------------------------------------------------------------ guards
def protected_set():
    return {x.strip() for x in env("OPS_PROTECTED_COLLECTIONS", "").split(",") if x.strip()}


def audit(a, action, detail):
    path = os.path.expanduser(env("OPS_AUDIT_LOG", "~/.milvus-ops-audit.log"))
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "uri": redact_uri(a.uri), "db": a.db,
                                "action": action, **detail}, ensure_ascii=False) + "\n")
    except OSError:
        pass


def guard(c, a, plan):
    """Return True if the action should execute; print the plan otherwise."""
    col = a.collection
    if col in protected_set():
        sys.exit(f"REFUSED: '{col}' is a protected collection; {a.cmd} is not allowed on it.")
    allow = [x.strip() for x in env("OPS_WRITE_ALLOW_PREFIX", "").split(",") if x.strip()]
    if allow and not any(col.startswith(p) for p in allow):
        sys.exit(f"REFUSED: OPS_WRITE_ALLOW_PREFIX={','.join(allow)} does not allow writes to '{col}'.")
    if not c.has_collection(col):
        sys.exit(f"ERROR: collection '{col}' does not exist")
    destructive = a.cmd in DESTRUCTIVE
    if not a.execute:
        need = f"--execute --confirm {col}" if destructive else "--execute"
        print(f"[dry-run] {a.cmd}: {plan}\n          nothing changed. To run it: add {need}")
        return False
    if destructive and a.confirm != col:
        sys.exit(f"REFUSED: destructive command '{a.cmd}' needs --confirm {col} (exact collection name).")
    return True


def done(a, plan, **extra):
    audit(a, a.cmd, {"collection": a.collection, "plan": plan, **extra})
    print(f"[done] {a.cmd}: {plan}" + (f" {json.dumps(extra, ensure_ascii=False)}" if extra else ""))


def row_count(c, col):
    """count(*) when loaded (exact, includes growing segments), otherwise flushed stats."""
    try:
        if "Loaded" in load_state(c, col):
            return c.query(col, filter="", output_fields=["count(*)"])[0]["count(*)"]
    except Exception:
        pass
    return c.get_collection_stats(col).get("row_count")


def load_state(c, col):
    return str(c.get_load_state(col).get("state")).replace("LoadState.", "")


# ------------------------------------------------------------------ READ
def metrics_url(a):
    if a.metrics_url:
        return a.metrics_url
    host = urllib.parse.urlparse(a.uri if "://" in a.uri else "http://" + a.uri).hostname or "localhost"
    return f"http://{host}:9091/metrics"


def http_get(url, timeout=5):
    with urlopen(url, timeout) as r:
        return r.status, r.read().decode(errors="replace")


def cmd_health(c, a):
    import pymilvus
    t0 = time.time()
    info = c.get_server_version(detail=True)
    rtt = round((time.time() - t0) * 1000, 1)
    out = {"uri": redact_uri(a.uri), "db": a.db, "server": info, "pymilvus": pymilvus.__version__, "rpc_ms": rtt}
    try:
        out["server_type"] = c.get_server_type()
    except Exception as e:
        out["server_type"] = f"n/a ({type(e).__name__})"
    out["databases"] = c.list_databases()
    out["collections"] = len(c.list_collections())
    url = metrics_url(a).rsplit("/", 1)[0] + "/healthz"
    try:
        st, body = http_get(url)
        out["healthz"] = {"url": url, "status": st, "body": body.strip()[:100]}
    except Exception as e:
        out["healthz"] = {"url": url, "error": str(e)[:120], "hint": "set MILVUS_METRICS_URL if the HTTP port is not 9091"}
    print_json(out)


def index_summary(c, col):
    out = []
    for name in c.list_indexes(col):
        try:
            d = c.describe_index(col, name)
        except Exception as e:
            out.append({"index_name": name, "error": str(e)[:80]}); continue
        out.append({k: d.get(k) for k in ("index_name", "field_name", "index_type", "metric_type", "state",
                                           "indexed_rows", "total_rows", "pending_index_rows") if k in d})
    return out


def cmd_collections(c, a):
    rows = []
    for col in sorted(c.list_collections()):
        r = {"collection": col}
        try:
            r["row_count"] = row_count(c, col)
            r["load_state"] = load_state(c, col)
            idx = index_summary(c, col)
            r["indexes"] = [f"{i.get('field_name')}:{i.get('index_type')}/{i.get('metric_type') or '-'}" for i in idx]
            r["protected"] = col in protected_set()
        except Exception as e:
            r["error"] = str(e)[:120]
        rows.append(r)
    if a.json:
        print_json(rows); return
    for r in rows:
        flag = " [protected]" if r.get("protected") else ""
        print(f"{r['collection']}{flag}: rows={r.get('row_count')} load={r.get('load_state')} "
              f"indexes={', '.join(r.get('indexes', [])) or '-'}" + (f" error={r['error']}" if "error" in r else ""))
    print(f"total: {len(rows)} collections")


def dtype_name(t):
    from pymilvus import DataType
    try:
        return DataType(int(t)).name if not isinstance(t, DataType) else t.name
    except (ValueError, TypeError):
        return str(t).replace("DataType.", "")


def cmd_describe(c, a):
    d = c.describe_collection(a.collection)
    fields = []
    for f in d["fields"]:
        x = {"name": f["name"], "type": dtype_name(f["type"])}
        x.update({k: v for k, v in (f.get("params") or {}).items()})
        for k in ("is_primary", "auto_id", "is_function_output", "nullable", "default_value"):
            if f.get(k):
                x[k] = f[k]
        fields.append(x)
    print_json({"collection": a.collection, "description": d.get("description"), "fields": fields,
                "functions": [{"name": f.get("name"), "type": str(f.get("type")), "input": f.get("input_field_names"),
                               "output": f.get("output_field_names"), "params": f.get("params")} for f in d.get("functions", [])],
                "enable_dynamic_field": d.get("enable_dynamic_field"), "num_shards": d.get("num_shards"),
                "consistency_level": d.get("consistency_level"), "properties": d.get("properties"),
                "partitions": c.list_partitions(a.collection), "indexes": index_summary(c, a.collection),
                "load_state": load_state(c, a.collection),
                "row_count": row_count(c, a.collection),
                "aliases": c.list_aliases(a.collection).get("aliases")})


def cmd_partitions(c, a):
    out = []
    for p in c.list_partitions(a.collection):
        try:
            rc = c.get_partition_stats(a.collection, p).get("row_count")
        except Exception:
            rc = None
        out.append({"partition": p, "row_count": rc})
    print_json(out)


def _seg(s):
    d = {}
    for k in ("segment_id", "collection_id", "partition_id", "num_rows", "state", "level", "is_sorted",
              "mem_size", "index_name", "node_ids", "storage_version"):
        v = getattr(s, k, None)
        if v is not None:
            d[k] = str(v) if k in ("state", "level") else v
    return d


def cmd_segments(c, a):
    pers = [_seg(s) for s in c.list_persistent_segments(a.collection)]
    try:
        loaded = [_seg(s) for s in c.list_loaded_segments(a.collection)]
    except Exception as e:
        loaded = f"n/a ({str(e)[:80]})"
    print_json({"collection": a.collection, "persistent_count": len(pers),
                "persistent_rows": sum(s.get("num_rows", 0) for s in pers), "persistent": pers,
                "loaded": loaded})


def cmd_compaction_state(c, a):
    print_json({"job_id": a.job_id, "state": str(c.get_compaction_state(a.job_id))})


CURATED = [  # (metric, group-by label or None, kind)
    ("milvus_num_node", "role_name", "sum"),
    ("milvus_rootcoord_collection_num", None, "sum"), ("milvus_rootcoord_partition_num", None, "sum"),
    ("milvus_rootcoord_entity_num", None, "sum"), ("milvus_rootcoord_credential_num", None, "sum"),
    ("milvus_rootcoord_num_of_roles", None, "sum"), ("milvus_rootcoord_force_deny_writing_counter", None, "sum"),
    ("milvus_querynode_collection_num", None, "sum"), ("milvus_querynode_entity_num", None, "sum"),
    ("milvus_querynode_segment_num", "segment_state", "sum"), ("milvus_querynode_disk_used_size", None, "sum"),
    ("milvus_datacoord_segment_num", "segment_state", "sum"), ("milvus_datacoord_stored_rows_num", None, "sum"),
    ("milvus_datacoord_stored_binlog_size", None, "sum"), ("milvus_datacoord_stored_index_files_size", None, "sum"),
    ("milvus_datacoord_compaction_task_num", "compaction_task_state", "sum"),
    ("milvus_proxy_req", "status", "sum"),
    ("milvus_proxy_sq_latency", "query_type", "avg"), ("milvus_proxy_mutation_latency", "msg_type", "avg"),
    ("process_resident_memory_bytes", None, "sum"), ("process_cpu_seconds_total", None, "sum"),
    ("milvus_jemalloc_resident_bytes", None, "sum"), ("go_goroutines", None, "sum"),
]
LINE = re.compile(r'^([a-zA-Z_:][\w:]*)(\{(.*)\})?\s+(\S+)')
LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def human(name, v):
    if name.endswith("bytes") or name.endswith("_size"):
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if abs(v) < 1024:
                return f"{v:.1f} {unit}"
            v /= 1024
    return round(v, 3) if isinstance(v, float) and not v.is_integer() else int(v)


def cmd_metrics(c, a):
    url = metrics_url(a)
    try:
        st, text = http_get(url, timeout=10)
    except Exception as e:
        sys.exit(f"metrics endpoint not reachable at {url}: {str(e)[:120]} (set MILVUS_METRICS_URL / --metrics-url)")
    samples = []
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = LINE.match(line)
        if m:
            try:
                samples.append((m.group(1), dict(LABEL.findall(m.group(3) or "")), float(m.group(4))))
            except ValueError:
                pass
    if a.grep:
        rx = re.compile(a.grep)
        for n, l, v in samples:
            if rx.search(n):
                print(n, json.dumps(l, ensure_ascii=False), v)
        return
    out = {"url": url, "families": len({re.sub(r"_(bucket|sum|count|total)$", "", n) for n, _, _ in samples}),
           "samples": len(samples)}
    bi = next((l for n, l, _ in samples if n == "milvus_build_info"), None)
    if bi:
        out["build_info"] = bi
    for name, by, kind in CURATED:
        if kind == "avg":
            s, cnt = {}, {}
            for n, l, v in samples:
                key = l.get(by, "all") if by else "all"
                if n == name + "_sum":
                    s[key] = s.get(key, 0) + v
                elif n == name + "_count":
                    cnt[key] = cnt.get(key, 0) + v
            if cnt:
                out[name + " (avg ms, count)"] = {k: (round(s.get(k, 0) / cnt[k], 2) if cnt[k] else None, int(cnt[k])) for k in cnt}
            continue
        vals = {}
        for n, l, v in samples:
            if n == name:
                key = l.get(by, "all") if by else "all"
                vals[key] = vals.get(key, 0) + v
        if vals:
            out[name] = human(name, vals["all"]) if list(vals) == ["all"] else {k: human(name, v) for k, v in vals.items()}
    print_json(out)


def cmd_users(c, a):
    out = {}
    try:
        users = c.list_users()
        out["users"] = []
        for u in users:
            try:
                d = c.describe_user(u)
                out["users"].append({"user": u, "roles": list(d.get("roles", []))})
            except Exception:
                out["users"].append({"user": u})
    except Exception as e:
        out["users"] = f"n/a: {str(e)[:150]} (authentication probably disabled)"
    try:
        out["roles"] = []
        for r in c.list_roles():
            try:
                privs = c.describe_role(r).get("privileges", [])
                out["roles"].append({"role": r, "privileges": [f"{p.get('object_type')}:{p.get('object_name')}:{p.get('privilege')}" for p in privs][:50]})
            except Exception:
                out["roles"].append({"role": r})
    except Exception as e:
        out["roles"] = f"n/a: {str(e)[:150]}"
    print_json(out)


BACKUP_GUIDE = """Milvus backup guidance (read-only advice, nothing is executed):
1. Logical backup with the official tool milvus-backup (https://github.com/zilliztech/milvus-backup):
     ./milvus-backup check                                   # verify milvus/minio config in configs/backup.yaml
     ./milvus-backup create -n bk_YYYYMMDD [-c col1,col2]    # snapshot collections (metadata + binlogs)
     ./milvus-backup list ; ./milvus-backup get -n bk_YYYYMMDD
     ./milvus-backup restore -n bk_YYYYMMDD -s _restored     # restore under a suffix, never over live data
   Run `flush COL --execute` first so growing segments are sealed into the backup.
2. Milvus 2.6+/3.x also offer collection snapshots (create_snapshot/restore_snapshot) – check the docs for your version.
3. Storage-level: back up the object storage bucket (MinIO/S3) + etcd/meta store consistently (stop writes or use
   storage snapshots). Prefer milvus-backup unless you know the layout.
4. Small KB collections: re-ingest from the source documents is a valid "backup" (keep the source files in git/S3).
5. Always test a restore into a new collection name and compare row counts (`collections`) before relying on it.
"""


def cmd_backup_guide(c, a):
    print(BACKUP_GUIDE)


# ------------------------------------------------------------------ WRITE / DESTRUCTIVE
def cmd_index_create(c, a):
    params = json.loads(a.params) if a.params else {}
    plan = f"create index on {a.collection}.{a.field} type={a.index_type} metric={a.metric or '-'} params={params}"
    existing = [i for i in index_summary(c, a.collection) if i.get("field_name") == a.field] if c.has_collection(a.collection) else []
    if existing:
        sys.exit(f"ERROR: field '{a.field}' already has index {existing[0].get('index_name')}; drop it first (index-drop).")
    if not guard(c, a, plan):
        return
    ip = c.prepare_index_params()
    kw = {"index_type": a.index_type, "params": params}
    if a.metric:
        kw["metric_type"] = a.metric
    if a.index_name:
        kw["index_name"] = a.index_name
    ip.add_index(a.field, **kw)
    c.create_index(a.collection, ip)
    done(a, plan)


def cmd_index_drop(c, a):
    st = load_state(c, a.collection) if c.has_collection(a.collection) else "?"
    plan = f"drop index '{a.index}' on {a.collection} (load_state={st}; Milvus requires the collection released first)"
    if not guard(c, a, plan):
        return
    if "NotLoad" not in st:
        sys.exit("ERROR: collection is loaded; run `release` first (also destructive, needs confirmation).")
    c.drop_index(a.collection, a.index)
    done(a, plan)


def cmd_load(c, a):
    plan = f"load {a.collection}" + (f" partitions={a.partitions}" if a.partitions else "")
    if not guard(c, a, plan):
        return
    if a.partitions:
        c.load_partitions(a.collection, a.partitions)
    else:
        c.load_collection(a.collection)
    done(a, plan, load_state=load_state(c, a.collection))


def cmd_release(c, a):
    plan = f"release {a.collection}" + (f" partitions={a.partitions}" if a.partitions else "") + " (searches will fail until loaded again)"
    if not guard(c, a, plan):
        return
    if a.partitions:
        c.release_partitions(a.collection, a.partitions)
    else:
        c.release_collection(a.collection)
    done(a, plan, load_state=load_state(c, a.collection))


def cmd_partition_create(c, a):
    plan = f"create partition '{a.partition}' in {a.collection}"
    if not guard(c, a, plan):
        return
    c.create_partition(a.collection, a.partition)
    done(a, plan)


def cmd_partition_drop(c, a):
    plan = f"drop partition '{a.partition}' from {a.collection} (all its data is deleted; must be released first)"
    if a.partition == "_default":
        sys.exit("REFUSED: the _default partition cannot be dropped")
    if not guard(c, a, plan):
        return
    if not c.has_partition(a.collection, a.partition):
        sys.exit(f"ERROR: partition '{a.partition}' not found")
    try:
        c.release_partitions(a.collection, [a.partition])
    except Exception:
        pass
    c.drop_partition(a.collection, a.partition)
    done(a, plan)


def cmd_flush(c, a):
    plan = f"flush {a.collection} (seal growing segments to object storage)"
    if not guard(c, a, plan):
        return
    c.flush(a.collection)
    done(a, plan)


def cmd_compact(c, a):
    plan = f"compact {a.collection}" + (" (clustering)" if a.clustering else "")
    if not guard(c, a, plan):
        return
    job = c.compact(a.collection, is_clustering=a.clustering)
    state = str(c.get_compaction_state(job))
    if a.wait:
        t0 = time.time()
        while "Completed" not in state and time.time() - t0 < a.wait_timeout:
            time.sleep(2); state = str(c.get_compaction_state(job))
    done(a, plan, job_id=job, state=state)


def cmd_delete_entities(c, a):
    load_ok = c.has_collection(a.collection) and "Loaded" in load_state(c, a.collection)
    n = None
    if load_ok:
        try:
            n = c.query(a.collection, filter=a.filter, output_fields=["count(*)"])[0]["count(*)"]
        except Exception as e:
            n = f"? ({str(e)[:80]})"
    plan = f"delete entities from {a.collection} where {a.filter!r} (matching now: {n if n is not None else 'unknown, collection not loaded'})"
    if not a.filter.strip():
        sys.exit("REFUSED: empty filter")
    if not guard(c, a, plan):
        return
    r = c.delete(a.collection, filter=a.filter)
    done(a, plan, delete_count=(r or {}).get("delete_count") if isinstance(r, dict) else None)


def cmd_drop_collection(c, a):
    rc = row_count(c, a.collection) if c.has_collection(a.collection) else "?"
    plan = f"DROP collection {a.collection} (row_count={rc}; irreversible – back it up first)"
    if not guard(c, a, plan):
        return
    c.drop_collection(a.collection)
    done(a, plan)


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def p(name, col=True, write=False):
        x = sub.add_parser(name)
        add_milvus_args(x)
        x.add_argument("--metrics-url", default=env("MILVUS_METRICS_URL"))
        if col:
            x.add_argument("collection")
        if write:
            x.add_argument("--execute", action="store_true", help="actually run it (default: dry-run)")
            x.add_argument("--confirm", default="", help="destructive ops: repeat the exact collection name")
        return x

    p("health", col=False)
    p("collections", col=False).add_argument("--json", action="store_true")
    p("describe"); p("partitions"); p("segments")
    x = p("compaction-state", col=False); x.add_argument("job_id", type=int)
    x = p("metrics", col=False); x.add_argument("--grep", help="print raw samples whose name matches REGEX")
    p("users", col=False); p("backup-guide", col=False)
    x = p("index-create", write=True)
    x.add_argument("--field", required=True); x.add_argument("--index-type", default="AUTOINDEX")
    x.add_argument("--metric"); x.add_argument("--params", help='JSON, e.g. {"M":16,"efConstruction":200}')
    x.add_argument("--index-name")
    x = p("index-drop", write=True); x.add_argument("--index", required=True)
    x = p("load", write=True); x.add_argument("--partitions", nargs="*")
    x = p("release", write=True); x.add_argument("--partitions", nargs="*")
    x = p("partition-create", write=True); x.add_argument("partition")
    x = p("partition-drop", write=True); x.add_argument("partition")
    p("flush", write=True)
    x = p("compact", write=True); x.add_argument("--clustering", action="store_true")
    x.add_argument("--wait", action="store_true"); x.add_argument("--wait-timeout", type=int, default=120)
    x = p("delete-entities", write=True); x.add_argument("--filter", required=True)
    p("drop-collection", write=True)
    a = ap.parse_args()
    c = connect(a)
    if getattr(a, "collection", None) and a.cmd not in WRITE | DESTRUCTIVE and not c.has_collection(a.collection):
        sys.exit(f"ERROR: collection '{a.collection}' does not exist")
    globals()["cmd_" + a.cmd.replace("-", "_")](c, a)


if __name__ == "__main__":
    main()
