#!/usr/bin/env python3
"""
Exporter aligned with Astro collections:

- Posts -> src/content/post/<slug>/index.md
  frontmatter: title, excerpt?, publishDate, updateDate?, draft, frontSlug, prev?, next?
- Projects -> src/content/projects/<slug>/index.md
  frontmatter: title, description, tier, image?, date?, links?, blog_posts?

Notes:
- No execution of notebooks (uses saved outputs only)
- Extracted assets from nbconvert are written next to index.md
- Multiple posts per repo supported (walks blog/**)
"""

import os, sys, subprocess, pathlib, json, textwrap, hashlib, shutil, re, yaml
from datetime import datetime, date
from typing import Dict, Any, Optional, Tuple, List
import nbformat
from nbconvert import MarkdownExporter

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "external"
POST_OUT = ROOT / "src" / "content" / "post"
PROJ_OUT = ROOT / "src" / "content" / "projects"
TEMPLATE_DIR = ROOT / "tools" / "templates"

def run(cmd, cwd=None):
    print("+", " ".join(cmd), "[cwd=" + str(cwd or ROOT) + "]")
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)

_slug_re = re.compile(r"[^a-z0-9-]+")
def slugify(s: str) -> str:
    return re.sub(r"-{2,}", "-", _slug_re.sub("-", s.lower()).strip("-"))

def natural_key(s: str):
    # "2" < "10", case-insensitive, preserves path segments
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def read_yaml(path: pathlib.Path) -> Dict[str, Any]:
    if path.exists():
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {}

def content_hash(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]

def summarize(text: str, max_chars=200) -> str:
    t = text.strip().splitlines()[0] if text.strip() else ""
    t = t.strip()
    return (t[: max_chars - 1] + "…") if len(t) > max_chars else t

def yaml_frontmatter_block(data: Dict[str, Any]) -> str:
    dumped = yaml.safe_dump(data, sort_keys=False, allow_unicode=True).rstrip()
    return f"---\n{dumped}\n---\n\n"

def parse_frontmatter(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    s = text.lstrip()
    if not s.startswith("---\n") and not s.startswith("---\r\n"):
        return None, text
    lines = text.splitlines(keepends=True)
    if not lines or not lines[0].strip().startswith("---"):
        return None, text
    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return None, text
    fm_text = "".join(lines[1:end_idx])
    body = "".join(lines[end_idx+1:])
    fm = yaml.safe_load(fm_text) or {}
    return fm, body

def ensure_frontslug_in_frontmatter(md_path: pathlib.Path, slug: str) -> None:
    """Ensure 'frontSlug' exists (and old 'slug' is removed) in the frontmatter of md_path."""
    if not md_path.exists():
        return
    text = md_path.read_text(encoding="utf-8")
    fm, body = parse_frontmatter(text)
    if fm is None:
        fm = {}
    # Remove legacy key to avoid front-end confusion
    if "slug" in fm:
        fm.pop("slug", None)
    if fm.get("frontSlug") != slug:
        fm["frontSlug"] = slug
        md_path.write_text(yaml_frontmatter_block(fm) + body, encoding="utf-8")

# --- Git ---
def clone_or_update(name: str, url: str, ref: str|None) -> pathlib.Path:
    dest = EXTERNAL / name
    if dest.exists():
        run(["git", "fetch", "origin"], cwd=dest)
        run(["git", "checkout", ref or "main"], cwd=dest)
        try:
            run(["git", "reset", "--hard", f"origin/{ref or 'main'}"], cwd=dest)
        except subprocess.CalledProcessError:
            pass
    else:
        run(["git", "clone", "--depth", "1", "--branch", ref or "main", url, str(dest)])
    return dest

def git_last_commit_date(repo_root: pathlib.Path, path: pathlib.Path) -> Optional[date]:
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
def make_project_card(repo_dir: pathlib.Path, blog_posts: Optional[List[Dict[str, str]]] = None) -> None:
    meta = read_yaml(repo_dir / "project.yml")
    title = meta.get("title") or repo_dir.name.replace("-", " ").title()
    slug = slugify(title)
    out_dir = PROJ_OUT / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    description = meta.get("description", "")
    tier = meta.get("tier", 2)
    image = meta.get("image") or meta.get("heroImage") or ""
    date_value = meta.get("date") or datetime.now().date()

    links = []
    if meta.get("repo_url"): links.append({"title": "Repository", "url": meta["repo_url"]})
    if meta.get("demo_url"): links.append({"title": "Demo", "url": meta["demo_url"]})

    fm: Dict[str, Any] = {
        "title": title,
        "description": description,
        "tier": tier,
        "image": image,
        "date": date_value,
        "links": links,
    }
    if blog_posts:
        fm["blog_posts"] = blog_posts

    (out_dir / "index.md").write_text(yaml_frontmatter_block(fm), encoding="utf-8")
    print(f"✓ project card {slug}" + (" with blog_posts" if blog_posts else ""))

# --- Posts ---
def export_notebook(ipynb: pathlib.Path, repo_name: str, rel_key: str, repo_dir: pathlib.Path) -> Dict[str, Any]:
    slug = slugify(f"{repo_name}-{rel_key}")
    out_dir = POST_OUT / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    h = content_hash(ipynb)
    marker = out_dir / f".hash-{h}"
    nb = nbformat.read(str(ipynb), as_version=4)

    title = nb.metadata.get("title") or f"{repo_name}: {ipynb.stem.replace('-', ' ').title()}"
    publish_date = git_last_commit_date(repo_dir, ipynb) or datetime.now().date()

    if marker.exists():
        ensure_frontslug_in_frontmatter(out_dir / "index.md", slug)
        print(f"= {slug} unchanged, skip")
        return {"title": title, "slug": slug, "publishDate": publish_date, "rel_key": rel_key}

    meta_desc = (nb.metadata.get("description") or nb.metadata.get("summary") or "")
    first_md = next((c.get("source", "") for c in nb.cells if c.get("cell_type") == "markdown"), "")
    excerpt = meta_desc or summarize(first_md, 200)

    exporter = MarkdownExporter(
        extra_template_basedirs=[str(TEMPLATE_DIR)],
        template_name="nb2astro",
        template_file="index.md.j2",
    )
    resources = {
        "metadata": {
            "title": title,
            "excerpt": excerpt,
            "publishDate": publish_date,
            "updateDate": None,
            "draft": False,
            "frontSlug": slug,  # ensure template has it
        }
    }

    body, res = exporter.from_notebook_node(nb, resources=resources)
    (out_dir / "index.md").write_text(body, encoding="utf-8")

    # enforce frontSlug even if the template doesn't emit it
    ensure_frontslug_in_frontmatter(out_dir / "index.md", slug)

    for name, data in (res.get("outputs") or {}).items():
        (out_dir / name).write_bytes(data)

    for old in out_dir.glob(".hash-*"): old.unlink()
    marker.touch()
    print(f"✓ exported notebook {slug}")
    return {"title": title, "slug": slug, "publishDate": publish_date, "rel_key": rel_key}

def export_markdown(md: pathlib.Path, repo_name: str, rel_key: str, repo_dir: pathlib.Path) -> Dict[str, Any]:
    slug = slugify(f"{repo_name}-{rel_key}")
    out_dir = POST_OUT / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    h = content_hash(md)
    marker = out_dir / f".hash-{h}"

    text = md.read_text(encoding="utf-8")
    fm, body = parse_frontmatter(text)

    title = (fm.get("title") if fm else md.stem.title())
    publishDate = (fm.get("publishDate") if fm else git_last_commit_date(repo_dir, md)) or datetime.now().date()

    if marker.exists():
        ensure_frontslug_in_frontmatter(out_dir / "index.md", slug)
        print(f"= {slug} unchanged, skip")
        return {"title": title, "slug": slug, "publishDate": publishDate, "rel_key": rel_key}

    if not fm:
        excerpt = summarize(text, 200)
        fm = {
            "title": title,
            "excerpt": excerpt,
            "publishDate": publishDate,
            "draft": False,
            "frontSlug": slug,  # add frontSlug when creating FM
        }
        text_to_write = yaml_frontmatter_block(fm) + body
    else:
        # Inject/overwrite frontSlug in existing FM while preserving other keys
        fm["frontSlug"] = slug
        fm.pop("slug", None)  # remove legacy key if present
        text_to_write = yaml_frontmatter_block(fm) + body

    (out_dir / "index.md").write_text(text_to_write, encoding="utf-8")

    for old in out_dir.glob(".hash-*"): old.unlink()
    marker.touch()
    print(f"✓ exported markdown {slug}")

    return {"title": title, "slug": slug, "publishDate": publishDate, "rel_key": rel_key}

def update_prev_next(posts_sorted: List[Dict[str, Any]]) -> None:
    """Write prev/next objects into each post's frontmatter based on natural filename order."""
    for i, p in enumerate(posts_sorted):
        post_md = POST_OUT / p["slug"] / "index.md"
        if not post_md.exists():  # safety
            continue
        text = post_md.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(text)
        if fm is None:
            fm = {}

        # always ensure frontSlug present and remove legacy key
        fm["frontSlug"] = p["slug"]
        fm.pop("slug", None)

        # prev
        if i > 0:
            prev = posts_sorted[i-1]
            fm["prev"] = {"title": prev["title"], "url": f"/blog/{prev['slug']}"}
        else:
            fm.pop("prev", None)
        # next
        if i < len(posts_sorted) - 1:
            nxt = posts_sorted[i+1]
            fm["next"] = {"title": nxt["title"], "url": f"/blog/{nxt['slug']}"}
        else:
            fm.pop("next", None)

        post_md.write_text(yaml_frontmatter_block(fm) + body, encoding="utf-8")
    if posts_sorted:
        print(f"✓ updated prev/next for {len(posts_sorted)} posts")

# --- Orchestration ---
def process_repo(name: str, url: str, ref: str|None) -> None:
    repo_dir = clone_or_update(name, url, ref)

    posts_info: List[Dict[str, Any]] = []

    blog_dir = repo_dir / "blog"
    if blog_dir.exists():
        for p in blog_dir.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(blog_dir).with_suffix("").as_posix()
            if p.suffix.lower() == ".ipynb":
                posts_info.append(export_notebook(p, name, rel, repo_dir))
            elif p.suffix.lower() == ".md":
                posts_info.append(export_markdown(p, name, rel, repo_dir))
    else:
        print(f"- no blog/ in {name}")

    # Order strictly by filename path (natural sort)
    posts_info.sort(key=lambda x: natural_key(x["rel_key"]))

    # Build ordered project list
    blog_posts = [{"title": p["title"], "url": f"/blog/{p['slug']}"} for p in posts_info]

    # Update prev/next pointers in each post
    update_prev_next(posts_info)

    # Write project card with ordered blog_posts
    make_project_card(repo_dir, blog_posts=blog_posts if blog_posts else None)

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
