# Usage: python3 scripts/validate_plugin.py .
# Checks: manifest keys/paths/semver, ${VAR} declarations, skill frontmatter, MCP tool names referenced in docs,
# scripts compile, uv script metadata, and every `milvus_kb.py <cmd>` / `milvus_ops.py <cmd>` referenced in
# skills/README really exists. The plugin is self-contained: no bots/ folder or generated prompts.
import json, os, re, glob, sys
root=os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "."); ok=True
def err(m):
    global ok; ok=False; print("FAIL:",m)
m=json.load(open(f"{root}/.cursor-plugin/plugin.json")); mcp=json.load(open(f"{root}/mcp.json"))
if not re.fullmatch(r"[a-z0-9]([a-z0-9.-]*[a-z0-9])?",m["name"]): err("name not kebab")
allowed={"name","displayName","publisher","category","tags","minClientVersions","description","version","author","homepage","repository","license","keywords","logo","rules","agents","skills","commands","hooks","mcpServers","variables"}
if set(m)-allowed: err(f"unknown manifest keys {set(m)-allowed}")
for k in ("logo","rules","agents","skills","commands","hooks","mcpServers"):
    for p in ([m[k]] if isinstance(m.get(k),str) else m.get(k) or []):
        if not isinstance(p,str) or p.startswith("http"): continue
        if p.startswith("/") or ".." in p.split("/"): err(f"{k} path not relative: {p}")
        if not os.path.exists(os.path.join(root,p)): err(f"{k} path missing: {p}")
used=set(re.findall(r"\$\{(\w+)\}",json.dumps(mcp))); declared=set(m["variables"]["properties"])
if used-declared: err(f"undeclared vars {used-declared}")
if declared-used: err(f"declared but unused {declared-used}")
okkw={"type","title","description","default","enum","const","properties","required","items","minLength","maxLength","minimum","maximum","pattern"}
for n,p in m["variables"]["properties"].items():
    if set(p)-okkw: err(f"var {n} bad keywords {set(p)-okkw}")
for sk in glob.glob(f"{root}/skills/*/SKILL.md"):
    t=open(sk).read(); fm=re.match(r"---\n(.*?)\n---\n",t,re.S)
    if not fm: err(f"no frontmatter {sk}"); continue
    name=re.search(r"^name:\s*(\S+)",fm.group(1),re.M); desc=re.search(r"^description:\s*(.+)",fm.group(1),re.M)
    if not name or name.group(1)!=os.path.basename(os.path.dirname(sk)): err(f"skill name mismatch {sk}")
    if not desc: err(f"no description {sk}")
# tool names referenced in skills/README must exist
tools={"milvus_text_search","milvus_list_collections","milvus_query","milvus_vector_search","milvus_hybrid_search","milvus_text_similarity_search","milvus_create_collection","milvus_insert_data","milvus_delete_entities","milvus_load_collection","milvus_release_collection","milvus_list_databases","milvus_use_database","milvus_get_collection_info"}
docs=glob.glob(f"{root}/skills/*/SKILL.md")+[f"{root}/README.md"]
for f in docs:
    refs=set(re.findall(r"`(milvus_[a-z_]+)`",open(f).read()))
    bad=refs-tools-{"milvus_upsert"}
    if bad: err(f"{f} references nonexistent tools {bad}")
if not re.fullmatch(r"\d+\.\d+\.\d+", m.get("version","")): err(f"version not semver: {m.get('version')}")
# scripts compile + uv inline metadata for runnable scripts
for py in glob.glob(f"{root}/scripts/*.py"):
    try: compile(open(py,encoding="utf-8").read(), py, "exec")
    except SyntaxError as e: err(f"compile {py}: {e}")
for py in ("milvus_kb.py","milvus_ops.py","e2e_test.py","smoke_test.py","make_sample_docs.py"):
    path=f"{root}/scripts/{py}"
    if not os.path.exists(path): err(f"missing script {py}"); continue
    if "# /// script" not in open(path).read(): err(f"{py} lacks uv inline metadata")
# subcommands referenced in docs exist
subs={}
for py in ("milvus_kb.py","milvus_ops.py"):
    src=open(f"{root}/scripts/{py}").read() if os.path.exists(f"{root}/scripts/{py}") else ""
    found=set(re.findall(r'(?:add_parser|\bp)\(\s*"([a-z][a-z-]*)"',src))
    for grp in re.findall(r'for name in \(([^)]*)\)',src): found|=set(re.findall(r'"([a-z-]+)"',grp))
    subs[py]=found
nref=0
for f in docs:
    for py,cmd in re.findall(r"(milvus_kb\.py|milvus_ops\.py)\s+([a-z][a-z-]+)",open(f).read()):
        nref+=1
        if cmd not in subs[py]: err(f"{os.path.relpath(f,root)} references unknown subcommand: {py} {cmd}")
# plugin-only repo: no bot configs / generated prompts, and no private LAN addresses in shipped files
if os.path.exists(f"{root}/bots"): err("bots/ folder must not be part of the plugin")
for f in glob.glob(f"{root}/**/*", recursive=True):
    if os.path.isdir(f) or "/.git/" in f or f.endswith((".pdf",".docx",".db",".svg")): continue
    try: t=open(f,encoding="utf-8").read()
    except (UnicodeDecodeError,OSError): continue
    if re.search(r"\b192\.168\.\d+\.\d+",t): err(f"private LAN address in {os.path.relpath(f,root)}")
    if re.search(r"render_bots|bots\.config\.json|SYSTEM_PROMPT",t) and not f.endswith("validate_plugin.py"): err(f"bot reference in {os.path.relpath(f,root)}")
print("skills:",len(glob.glob(f"{root}/skills/*/SKILL.md")),"vars used:",sorted(used),"version:",m.get("version"),
      "kb cmds:",len(subs.get("milvus_kb.py",())),"ops cmds:",len(subs.get("milvus_ops.py",())),"doc cmd refs checked:",nref)
print("VALID" if ok else "INVALID")
