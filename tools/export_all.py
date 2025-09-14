#!/usr/bin/env python3
"""
MVP exporter aligned with your Astro collections:

- Posts go to: src/content/post/<slug>/index.md
  frontmatter: title, excerpt?, publishDate, updateDate?, draft, tags
- Projects go to: src/content/projects/<slug>/index.md
  frontmatter: title, description, tier, image?, date?, links?, blog_posts?

Notes:
- NO execution of notebooks (uses saved outputs only)
- Extracted assets from nbconvert are written next to index.md
- Multiple posts per repo supported (walks blog/**)
"""

import os, sys, subprocess, pathlib, json, textwrap, hashlib, shutil, re, yaml
from datetime import datetime
from typing import Dict, Any
import nbformat
from nbconvert import MarkdownExporter

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "external"
POST_OUT = ROOT / "src" / "content" / "post"       # <— changed
PROJ_OUT = ROOT / "src" / "content" / "projects"
TEMPLATE_DIR = ROOT / "tools" / "templates"

def run(cmd, cwd=None):
    print("+", " ".join(cmd), "[cwd=" + str(cwd or ROOT) + "]")
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)

_slug_re = re.compile(r"[^a-z0-9-]+")
def slugify(s: str) -> str:
    return re.sub(r"-{2,}", "-", _slug_re.sub("-", s.lower()).strip("-"))

def read_yaml(path: pathlib.Path) -> Dict[str, Any]:
    if path.exists():
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {}

def content_hash(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]

def summarize(text: str, max_chars=200) -> str:
    # simple excerpt maker (first line/paragraph, trimmed)
    t = text.strip().splitlines()[0] if text.strip() else ""
    t = t.strip()
    return (t[: max_chars - 1] + "…") if len(t) > max_chars else t

# --- Git ---
def clone_or_update(name: str, url: str, ref: str|None) -> pathlib.Path:
    dest = EXTERNAL / name
    if dest.exists():
        run(["git", "fetch", "origin"], cwd=dest)
        run(["git", "checkout", ref or "main"], cwd=dest)
        try:
            run(["git", "reset", "--hard", f"origin/{ref or 'main'}"], cwd=dest)
        except subprocess.CalledProcessError:
            # tag or detached; ignore
            pass
    else:
        run(["git", "clone", "--depth", "1", "--branch", ref or "main", url, str(dest)])
    return dest

def git_last_commit_date(repo_root: pathlib.Path, path: pathlib.Path):
    """Return the last commit date as a datetime.date (no time component)."""
    try:
        out = subprocess.check_output(
            ["git", "log", "-1", "--format=%cI", "--", str(path)],
            cwd=str(repo_root)
        ).decode().strip()
        if out:
            return datetime.fromisoformat(out).date()
    except subprocess.CalledProcessError:
        return None


# --- Projects ---
def make_project_card(repo_dir: pathlib.Path) -> None:
    meta = read_yaml(repo_dir / "project.yml")
    title = meta.get("title") or repo_dir.name.replace("-", " ").title()
    slug = slugify(title)
    out_dir = PROJ_OUT / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    # Map to your schema
    description = meta.get("description", "")
    tier = meta.get("tier", 2)
    image = meta.get("image") or meta.get("heroImage") or ""  # allow legacy key
    date = meta.get("date") or datetime.now().date().isoformat()
    links = []
    if meta.get("repo_url"):
        links.append({"label": "Repository", "url": meta["repo_url"]})
    if meta.get("demo_url"):
        links.append({"label": "Demo", "url": meta["demo_url"]})

    body = textwrap.dedent(f"""\
    ---
    title: "{title}"
    description: "{description}"
    tier: {tier}
    image: "{image}"
    date: {date}
    links: {json.dumps(links)}
    # blog_posts: []  # fill later if you want to list related posts
    ---
    """)
    (out_dir / "index.md").write_text(body, encoding="utf-8")
    print(f"✓ project card {slug}")

# --- Posts ---
def export_notebook(ipynb: pathlib.Path, repo_name: str, rel_key: str, repo_dir: pathlib.Path) -> None:
    slug = slugify(f"{repo_name}-{rel_key}")
    out_dir = POST_OUT / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    h = content_hash(ipynb)
    marker = out_dir / f".hash-{h}"
    if marker.exists():
        print(f"= {slug} unchanged, skip")
        return

    nb = nbformat.read(str(ipynb), as_version=4)
    # Derive a reasonable excerpt:
    # prefer a 'description' in notebook metadata, else first markdown cell line
    meta_desc = (nb.metadata.get("description") or
                 nb.metadata.get("summary") or "")
    first_md = ""
    for cell in nb.cells:
        if cell.get("cell_type") == "markdown":
            first_md = cell.get("source", "")
            break
    excerpt = meta_desc or summarize(first_md, 200)

    # Use template with correct keys in frontmatter
    exporter = MarkdownExporter(
        extra_template_basedirs=[str(TEMPLATE_DIR)],
        template_name="nb2astro",
        template_file="index.md.j2",
    )
    # Supply frontmatter data via resources.metadata
    resources = {
        "metadata": {
            "title": nb.metadata.get("title") or f"{repo_name}: {ipynb.stem.replace('-', ' ').title()}",
            "excerpt": excerpt,
            "publishDate": git_last_commit_date(repo_dir, ipynb) or datetime.now().date().isoformat(),
            "updateDate": None,
            "draft": False,
            "tags": [f"project:{repo_name}"],
        }
    }

    body, res = exporter.from_notebook_node(nb, resources=resources)
    (out_dir / "index.md").write_text(body, encoding="utf-8")

    # Persist any extracted outputs (images, etc.)
    for name, data in (res.get("outputs") or {}).items():
        (out_dir / name).write_bytes(data)

    for old in out_dir.glob(".hash-*"): old.unlink()
    marker.touch()
    print(f"✓ exported notebook {slug}")

def export_markdown(md: pathlib.Path, repo_name: str, rel_key: str, repo_dir: pathlib.Path) -> None:
    slug = slugify(f"{repo_name}-{rel_key}")
    out_dir = POST_OUT / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    h = content_hash(md)
    marker = out_dir / f".hash-{h}"
    if marker.exists():
        print(f"= {slug} unchanged, skip")
        return

    text = md.read_text(encoding="utf-8")
    if not text.lstrip().startswith("---"):
        excerpt = summarize(text, 200)
        publishDate = git_last_commit_date(repo_dir, md) or datetime.now().date().isoformat()
        fm = f"""
---
title: "{md.stem.title()}"
excerpt: "{excerpt}"
publishDate: {publishDate}
draft: false
tags: ["project:{repo_name}"]
---
"""
        text = fm + "\n" + text

    (out_dir / "index.md").write_text(text, encoding="utf-8")

    for old in out_dir.glob(".hash-*"): old.unlink()
    marker.touch()
    print(f"✓ exported markdown {slug}")

# --- Orchestration ---
def process_repo(name: str, url: str, ref: str|None) -> None:
    repo_dir = clone_or_update(name, url, ref)
    make_project_card(repo_dir)

    blog_dir = repo_dir / "blog"
    if not blog_dir.exists():
        print(f"- no blog/ in {name}")
        return

    for p in blog_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(blog_dir).with_suffix("").as_posix()
        if p.suffix.lower() == ".ipynb":
            export_notebook(p, name, rel, repo_dir)
        elif p.suffix.lower() == ".md":
            export_markdown(p, name, rel, repo_dir)

def main():
    EXTERNAL.mkdir(exist_ok=True)
    POST_OUT.mkdir(parents=True, exist_ok=True)
    PROJ_OUT.mkdir(parents=True, exist_ok=True)

    manifest_path = ROOT / "content-manifest.yml"
    if not manifest_path.exists():
        print("ERROR: content-manifest.yml missing at repo root", file=sys.stderr)
        sys.exit(1)

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    for r in manifest.get("repos", []):
        process_repo(r["name"], r["url"], r.get("ref"))

if __name__ == "__main__":
    main()
