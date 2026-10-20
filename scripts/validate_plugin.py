"""Validate manifest, skills, and executable command references; exit nonzero on failure."""

import argparse
import importlib
import json
import re
import sys
from pathlib import Path


def validate(root):
    errors = []
    try:
        manifest = json.loads((root / ".cursor-plugin/plugin.json").read_text())
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", manifest.get("name", "")):
            errors.append("Invalid plugin name")
        if not re.fullmatch(r"\d+\.\d+\.\d+", manifest.get("version", "")):
            errors.append("Invalid semantic version")
        logo = Path(manifest.get("logo", ""))
        if logo.is_absolute() or ".." in logo.parts or not (root / logo).is_file():
            errors.append("Missing or invalid logo path")
    except (OSError, ValueError) as exc:
        errors.append(f"Invalid manifest: {exc}")
    if not (root / "pyproject.toml").is_file() or not (root / "uv.lock").is_file():
        errors.append("Missing project dependencies or lock file")
    if (root / "mcp.json").exists():
        errors.append("Unexpected background server configuration")
    skills = sorted(root.glob("skills/*/SKILL.md"))
    if not skills:
        errors.append("No skills found")
    docs = [root / "README.md", *skills]
    for skill in skills:
        frontmatter = re.match(r"---\n(.*?)\n---\n", skill.read_text(), re.S)
        if not frontmatter:
            errors.append(f"Missing frontmatter: {skill}")
            continue
        name = re.search(r"^name:\s*(\S+)", frontmatter[1], re.M)
        description = re.search(r"^description:\s*(.+)", frontmatter[1], re.M)
        if not name or name[1] != skill.parent.name or not description:
            errors.append(f"Invalid skill metadata: {skill}")
    commands = {}
    sys.path.insert(0, str(root / "scripts"))
    for name in ("milvus_kb", "milvus_inspect"):
        try:
            module = importlib.import_module(name)
            parser = module.build_parser()
            commands[name + ".py"] = next(
                set(a.choices) for a in parser._actions if isinstance(a, argparse._SubParsersAction)
            )
        except Exception as exc:
            errors.append(f"Cannot load {name}: {exc}")
    references = 0
    for doc in docs:
        try:
            content = doc.read_text()
        except OSError as exc:
            errors.append(str(exc))
            continue
        for script, command in re.findall(r'(milvus_[a-z_]+\.py)["\x27]?\s+([a-z][a-z-]+)', content):
            references += 1
            if command not in commands.get(script, set()):
                errors.append(f"{doc}: unknown command {script} {command}")
    for path in (root / "scripts").glob("*.py"):
        try:
            compile(path.read_text(), str(path), "exec")
        except SyntaxError as exc:
            errors.append(str(exc))
    for error in errors:
        print("FAIL:", error)
    print(f"skills={len(skills)} command_references={references}")
    print("INVALID" if errors else "VALID")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(
        validate(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1])
    )
