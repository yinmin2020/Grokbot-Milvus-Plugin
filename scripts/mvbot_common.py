"""Shared helpers for the knowledge-base and ops scripts (milvus_kb.py, milvus_ops.py).

Configuration comes from CLI flags first, then environment variables:
  MILVUS_URI (default http://localhost:19530), MILVUS_TOKEN, MILVUS_DB (default "default")
  EMBED_PROVIDER  ollama | openai            (default ollama)
  EMBED_ENDPOINT  Ollama: http://localhost:11434 ; OpenAI-compatible: https://api.openai.com/v1
  EMBED_MODEL     e.g. embeddinggemma-2:740m / text-embedding-3-small
  EMBED_API_KEY   only for the openai provider (falls back to OPENAI_API_KEY); never printed
  KB_COLLECTION   default knowledge-base collection (default kb_docs)
  OPS_PROTECTED_COLLECTIONS  comma-separated collections the ops script must never change
"""
import ipaddress, json, os, sys, time, urllib.parse, urllib.request, urllib.error


def _is_private(url):
    host = urllib.parse.urlparse(url).hostname or ""
    if host in ("localhost",) or host.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_private or ip.is_loopback
    except ValueError:
        return False


def urlopen(req, timeout):
    """urlopen that bypasses system/env HTTP proxies for LAN/loopback hosts (macOS system proxies
    otherwise turn requests to private LAN addresses or localhost into 502s)."""
    url = req.full_url if isinstance(req, urllib.request.Request) else req
    if _is_private(url) and env("MVBOT_USE_PROXY_FOR_LAN", "0") != "1":
        return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout)
    return urllib.request.urlopen(req, timeout=timeout)


def env(name, default=None):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def add_milvus_args(ap):
    g = ap.add_argument_group("Milvus connection")
    g.add_argument("--uri", default=env("MILVUS_URI", "http://localhost:19530"))
    g.add_argument("--token", default=env("MILVUS_TOKEN", ""), help="user:password or API key (or env MILVUS_TOKEN)")
    g.add_argument("--db", default=env("MILVUS_DB", "default"))


def connect(a):
    os.environ.pop("MILVUS_URI", None)  # pymilvus validates this env var at import time
    from pymilvus import MilvusClient
    kw = {"uri": a.uri}
    if a.token:
        kw["token"] = a.token
    if a.db and a.db != "default":
        kw["db_name"] = a.db
    return MilvusClient(**kw)


def redact_uri(uri):
    # strip credentials if someone put user:pass@ in the URI
    if "@" in uri and "://" in uri:
        scheme, rest = uri.split("://", 1)
        return scheme + "://***@" + rest.split("@", 1)[1]
    return uri


# ---------------------------------------------------------------- embeddings
def add_embed_args(ap):
    g = ap.add_argument_group("Embedding")
    g.add_argument("--embed-provider", choices=["ollama", "openai"], default=env("EMBED_PROVIDER"))
    g.add_argument("--embed-endpoint", default=env("EMBED_ENDPOINT"))
    g.add_argument("--embed-model", default=env("EMBED_MODEL"))
    g.add_argument("--embed-batch", type=int, default=int(env("EMBED_BATCH", "16")))
    g.add_argument("--embed-timeout", type=float, default=float(env("EMBED_TIMEOUT", "120")))


# Some embedding models are trained with task prefixes. They are recorded in the
# collection description so queries always use the same convention as ingest.
# min_score = suggested "found" threshold on the best dense cosine (calibrate per model with off-topic questions).
MODEL_PREFIXES = {
    "embeddinggemma": {"query": "task: search result | query: ", "doc": "title: {title} | text: ", "min_score": 0.62},
    "nomic-embed-text": {"query": "search_query: ", "doc": "search_document: ", "min_score": 0.5},
}


def default_prefixes(model):
    m = (model or "").lower()
    for key, p in MODEL_PREFIXES.items():
        if key in m:
            return dict(p)
    return {"query": "", "doc": "", "min_score": 0.5}


class Embedder:
    def __init__(self, provider, endpoint, model, batch=16, timeout=120.0):
        self.provider = provider or "ollama"
        if self.provider == "ollama":
            self.endpoint = (endpoint or "http://localhost:11434").rstrip("/")
        else:
            self.endpoint = (endpoint or "https://api.openai.com/v1").rstrip("/")
        if not model:
            sys.exit("ERROR: no embedding model. Pass --embed-model or set EMBED_MODEL.")
        self.model, self.batch, self.timeout = model, max(1, batch), timeout
        self.api_key = env("EMBED_API_KEY") or env("OPENAI_API_KEY") or ""

    def _post(self, url, body):
        headers = {"Content-Type": "application/json"}
        if self.provider == "openai" and self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
        for attempt in range(3):
            try:
                with urlopen(req, self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="replace")[:300]
                if e.code >= 500 and attempt < 2:
                    time.sleep(1 + attempt)
                    continue
                sys.exit(f"ERROR: embedding request failed HTTP {e.code} at {url}: {msg}")
            except urllib.error.URLError as e:
                if attempt < 2:
                    time.sleep(1 + attempt)
                    continue
                sys.exit(f"ERROR: cannot reach embedding endpoint {url}: {e.reason}")

    def embed(self, texts):
        out = []
        for i in range(0, len(texts), self.batch):
            chunk = texts[i:i + self.batch]
            if self.provider == "ollama":
                d = self._post(self.endpoint + "/api/embed", {"model": self.model, "input": chunk})
                vecs = d.get("embeddings")
            else:
                d = self._post(self.endpoint + "/embeddings", {"model": self.model, "input": chunk})
                vecs = [x["embedding"] for x in sorted(d.get("data", []), key=lambda x: x.get("index", 0))]
            if not vecs or len(vecs) != len(chunk):
                sys.exit(f"ERROR: embedding endpoint returned {len(vecs or [])} vectors for {len(chunk)} inputs")
            out.extend(vecs)
        return out

    def dim(self):
        return len(self.embed(["dimension probe"])[0])


def print_json(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))
