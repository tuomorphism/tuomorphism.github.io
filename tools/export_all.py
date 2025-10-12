#!/usr/bin/env python3
"""
Minimal exporter for Astro collections with frontSlug + prev/next.

- Posts -> src/content/post/<slug>/index.md
  frontmatter: title, publishDate, frontSlug, prev?, next?
- Projects -> src/content/projects/<slug>/index.md
  frontmatter: title, description, tier, image?, date?, links?, blog_posts?

Notes:
- No notebook execution (uses saved outputs only)
- Extracted nbconvert outputs are written next to index.md
- Copies user-referenced assets (images/videos/files) into ./assets/ and rewrites links
- Adds anchor IDs to markdown headings and exposes ToC items to the template
- Adds prev/next pointers after all posts are written
"""

import sys
import subprocess
import pathlib
import hashlib
import re
from datetime import datetime, date
from typing import Dict, Any, Optional, Tuple, List

import yaml
import nbformat
from nbconvert import MarkdownExporter

# Paths
ROOT = pathlib.Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "external"
POST_OUT = ROOT / "src" / "content" / "post"
PROJ_OUT = ROOT / "src" / "content" / "projects"
TEMPLATE_DIR = ROOT / "tools" / "templates"

# Config
ASSET_DIR_NAME = "assets"  # per-post folder under the exported post
ASSET_SOURCE_DIR_CANDIDATES = ("assets", "_assets")  # relative search roots under blog/
MAX_TOC_DEPTH = 3          # headings deeper than this are ignored

# Regex helpers
_MD_LINK_IMG = re.compile(r'(!?)\[(?P<alt>[^\]]*)\]\((?P<url>[^)\s]+)(?:\s+"[^"]*")?\)')
_HTML_SRC_OR_HREF = re.compile(r'(?P<attr>\bsrc\b|\bhref\b)\s*=\s*([\'"])(?P<url>[^\'"]+)\2')
_MD_HEADING = re.compile(r'^(?P<hash>#{1,6})\s+(?P<text>.+?)\s*$', re.MULTILINE)
_slug_re = re.compile(r"[^a-z0-9-]+")


# ---------- Utilities

def run(cmd, cwd=None):
    print("+", " ".join(cmd), "[cwd=" + str(cwd or ROOT) + "]")
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)

def slugify(s: str) -> str:
    return re.sub(r"-{2,}", "-", _slug_re.sub("-", s.lower()).strip("-"))

def natural_key(s: str):
    # natural sort preserving path segments: "2" < "10"
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s.lower())]

def read_yaml(path: pathlib.Path) -> Dict[str, Any]:
    if path.exists():
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {}

def content_hash(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]

def yaml_frontmatter_block(data: Dict[str, Any]) -> str:
    dumped = yaml.safe_dump(data, sort_keys=False, allow_unicode=True).rstrip()
    return f"---\n{dumped}\n---\n\n"

def parse_frontmatter(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    s = text.lstrip()
    if not s.startswith("---\n") and not s.startswith("---\r\n"):
        return None, text
    lines = text.splitlines(keepends=True)
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            fm_text = "".join(lines[1:i])
            body = "".join(lines[i+1:])
            fm = yaml.safe_load(fm_text) or {}
            return fm, body
    return None, text

def ensure_frontslug_in_frontmatter(md_path: pathlib.Path, slug: str) -> None:
    """Guarantee frontSlug exists (and is correct) in the file's frontmatter."""
    if not md_path.exists():
        return
    text = md_path.read_text(encoding="utf-8")
    fm, body = parse_frontmatter(text)
    fm = (fm or {})
    if fm.get("frontSlug") != slug:
        fm["frontSlug"] = slug
        md_path.write_text(yaml_frontmatter_block(fm) + body, encoding="utf-8")


# ---------- Git

def clone_or_update(name: str, url: str, ref: str | None) -> pathlib.Path:
    dest = EXTERNAL / name
    if dest.exists():
        run(["git", "fetch", "origin", "--prune"], cwd=dest)
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


# ---------- Asset + ToC helpers

def is_relative_local(url: str) -> bool:
    if not url:
        return False
    if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', url):  # http(s), etc.
        return False
    if url.startswith(("data:", "#", "/")):
        return False
    return True

def ensure_dir(p: pathlib.Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def copy_asset_make_name(src: pathlib.Path, out_assets_dir: pathlib.Path) -> str:
    """Copy asset to out_assets_dir with a short content hash to avoid collisions."""
    data = src.read_bytes()
    h = hashlib.sha256(data).hexdigest()[:8]
    stem, ext = src.stem, src.suffix
    safe_stem = slugify(stem) or "asset"
    fname = f"{safe_stem}.{h}{ext}"
    ensure_dir(out_assets_dir)
    (out_assets_dir / fname).write_bytes(data)
    return f"{ASSET_DIR_NAME}/{fname}"

def resolve_asset_candidate(base_dir: pathlib.Path, url: str) -> Optional[pathlib.Path]:
    """Resolve a relative url to an existing file, trying common asset roots."""
    cand = (base_dir / url).resolve()
    if cand.exists() and cand.is_file():
        return cand
    for adir in ASSET_SOURCE_DIR_CANDIDATES:
        cand2 = (base_dir / adir / url).resolve()
        if cand2.exists() and cand2.is_file():
            return cand2
    return None

def rewrite_urls_and_copy_assets(text: str, base_dir: pathlib.Path, out_assets_dir: pathlib.Path) -> str:
    """Rewrite markdown/HTML URLs to ./assets/* and copy the files."""

    def _md_repl(m: re.Match) -> str:
        bang = m.group(1)  # '!' for image or '' for link
        alt = m.group('alt')
        url = m.group('url')
        if is_relative_local(url):
            src = resolve_asset_candidate(base_dir, url)
            if src:
                rel = copy_asset_make_name(src, out_assets_dir)
                return f"{bang}[{alt}]({rel})"
        return m.group(0)

    def _html_repl(m: re.Match) -> str:
        attr = m.group('attr')
        url = m.group('url')
        if is_relative_local(url):
            src = resolve_asset_candidate(base_dir, url)
            if src:
                rel = copy_asset_make_name(src, out_assets_dir)
                return f'{attr}="{rel}"'
        return m.group(0)

    text = _MD_LINK_IMG.sub(_md_repl, text)
    text = _HTML_SRC_OR_HREF.sub(_html_repl, text)
    return text

def slugify_heading(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r'\s+', '-', s)
    s = re.sub(r'[^a-z0-9\-]', '', s)
    s = re.sub(r'-{2,}', '-', s).strip('-')
    return s or "section"

def add_ids_and_collect_toc(md_text: str, max_depth: int = 3) -> Tuple[str, List[Dict[str, Any]]]:
    """Inject explicit <a id="…"></a> anchors and build a flat ToC list."""
    used_ids: Dict[str, int] = {}
    toc: List[Dict[str, Any]] = []

    def unique_id(base: str) -> str:
        n = used_ids.get(base, 0)
        used_ids[base] = n + 1
        return base if n == 0 else f"{base}-{n}"

    lines = md_text.splitlines()
    for i, line in enumerate(lines):
        m = _MD_HEADING.match(line)
        if not m:
            continue
        level = len(m.group('hash'))
        text = m.group('text').strip()
        hid = unique_id(slugify_heading(text))
        # Ensure linkability regardless of markdown engine
        lines[i] = f'<a id="{hid}"></a>\n{m.group("hash")} {text}'
        if level <= max_depth:
            toc.append({"level": level, "text": text, "id": hid})
    return "\n".join(lines) + ("\n" if not md_text.endswith("\n") else ""), toc


# ---------- Projects

def _merge_and_dedupe_links(links: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    deduped: List[Dict[str, str]] = []
    for li in links:
        if not isinstance(li, dict):
            continue
        url = li.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append({"title": li.get("title") or "Link", "url": url})
    return deduped

def make_project_card(
    repo_dir: pathlib.Path,
    repo_url: Optional[str],
    blog_posts: Optional[List[Dict[str, str]]] = None
) -> None:
    meta = read_yaml(repo_dir / "project.yml")
    title = meta.get("title") or repo_dir.name.replace("-", " ").title()
    slug = slugify(title)
    out_dir = PROJ_OUT / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    description = meta.get("description", "")
    tier = meta.get("tier", 2)
    image = meta.get("image") or meta.get("heroImage") or ""
    date_value = meta.get("date") or datetime.now().date()

    links: List[Dict[str, str]] = []
    if repo_url:
        links.append({"title": "Repo", "url": repo_url})
    for li in meta.get("links", []) or []:
        if isinstance(li, dict) and li.get("url"):
            links.append({"title": li.get("title") or "Link", "url": li["url"]})
    if meta.get("repo_url"):
        links.append({"title": "Repo", "url": meta["repo_url"]})
    if meta.get("demo_url"):
        links.append({"title": "Demo", "url": meta["demo_url"]})
    links = _merge_and_dedupe_links(links)

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


# ---------- Posts

def export_notebook(ipynb: pathlib.Path, repo_name: str, rel_key: str, repo_dir: pathlib.Path) -> Dict[str, Any]:
    slug = slugify(f"{repo_name}-{rel_key}")
    out_dir = POST_OUT / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_assets_dir = out_dir / ASSET_DIR_NAME

    marker = out_dir / f".hash-{content_hash(ipynb)}"
    nb = nbformat.read(str(ipynb), as_version=4)

    publish_date = git_last_commit_date(repo_dir, ipynb) or datetime.now().date()

    # Preprocess markdown cells: rewrite asset URLs, then add anchors & collect ToC
    base_dir = ipynb.parent
    md_chunks: List[str] = []
    for cell in nb.cells:
        if cell.get("cell_type") == "markdown":
            text = rewrite_urls_and_copy_assets(cell.get("source", ""), base_dir, out_assets_dir)
            md_chunks.append(text)
            cell["source"] = text

    toc_items: List[Dict[str, Any]] = []
    if md_chunks:
        full_md = "\n\n".join(md_chunks)
        anchored_full, toc_items = add_ids_and_collect_toc(full_md, max_depth=MAX_TOC_DEPTH)
        new_chunks = anchored_full.split("\n\n")
        it = iter(new_chunks)
        for cell in nb.cells:
            if cell.get("cell_type") == "markdown":
                try:
                    cell["source"] = next(it)
                except StopIteration:
                    break
    
    first_h1 = None
    h1_re = re.compile(r'^\s*#\s+(.+)$', re.MULTILINE)
    for chunk in md_chunks:
        m = h1_re.search(chunk)
        if m:
            first_h1 = m.group(1).strip()
            break
    
    title = nb.metadata.get("title") or first_h1 or f"{repo_name}: {ipynb.stem.replace('-', ' ').title()}"
    
    if marker.exists():
        ensure_frontslug_in_frontmatter(out_dir / "index.md", slug)
        print(f"= {slug} unchanged, skip")
        return {"title": title, "slug": slug, "publishDate": publish_date, "rel_key": rel_key}

    exporter = MarkdownExporter(
        extra_template_basedirs=[str(TEMPLATE_DIR)],
        template_name="nb2astro",
        template_file="index.md.j2",
    )
    resources = {
        "metadata": {
            "title": title,
            "publishDate": publish_date,
            "frontSlug": slug,
            "toc_items": toc_items,
        }
    }

    body, res = exporter.from_notebook_node(nb, resources=resources)
    (out_dir / "index.md").write_text(body, encoding="utf-8")
    ensure_frontslug_in_frontmatter(out_dir / "index.md", slug)

    for name, data in (res.get("outputs") or {}).items():
        (out_dir / name).write_bytes(data)

    for old in out_dir.glob(".hash-*"):
        old.unlink()
    marker.touch()
    print(f"✓ exported notebook {slug}")
    return {"title": title, "slug": slug, "publishDate": publish_date, "rel_key": rel_key}

def export_markdown(md: pathlib.Path, repo_name: str, rel_key: str, repo_dir: pathlib.Path) -> Dict[str, Any]:
    slug = slugify(f"{repo_name}-{rel_key}")
    out_dir = POST_OUT / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_assets_dir = out_dir / ASSET_DIR_NAME

    marker = out_dir / f".hash-{content_hash(md)}"
    text = md.read_text(encoding="utf-8")
    fm, body = parse_frontmatter(text)

    # Rewrite/copy assets and add anchors/ToC
    base_dir = md.parent
    body = rewrite_urls_and_copy_assets(body, base_dir, out_assets_dir)
    body, toc_items = add_ids_and_collect_toc(body, max_depth=MAX_TOC_DEPTH)

    title = (fm.get("title") if fm else md.stem.title())
    publishDate = (fm.get("publishDate") if fm else git_last_commit_date(repo_dir, md)) or datetime.now().date()

    if marker.exists():
        ensure_frontslug_in_frontmatter(out_dir / "index.md", slug)
        print(f"= {slug} unchanged, skip")
        return {"title": title, "slug": slug, "publishDate": publishDate, "rel_key": rel_key}

    # Minimal frontmatter with enforced frontSlug
    if not fm:
        fm = {"title": title, "publishDate": publishDate, "frontSlug": slug}
    else:
        fm.setdefault("title", title)
        fm.setdefault("publishDate", publishDate)
        fm["frontSlug"] = slug

    (out_dir / "index.md").write_text(yaml_frontmatter_block(fm) + body, encoding="utf-8")
    ensure_frontslug_in_frontmatter(out_dir / "index.md", slug)

    for old in out_dir.glob(".hash-*"):
        old.unlink()
    marker.touch()
    print(f"✓ exported markdown {slug}")

    return {"title": title, "slug": slug, "publishDate": publishDate, "rel_key": rel_key}


def update_prev_next(posts_sorted: List[Dict[str, Any]]) -> None:
    """
    Inject prev/next pointers into each post's frontmatter based on natural path order.
    Writes:
      prev: { title, url }
      next: { title, url }
    Keeps frontSlug intact.
    """
    for i, p in enumerate(posts_sorted):
        post_md = POST_OUT / p["slug"] / "index.md"
        if not post_md.exists():
            continue
        text = post_md.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(text)
        fm = (fm or {})

        # keep required key stable
        fm["frontSlug"] = p["slug"]

        # prev
        if i > 0:
            prv = posts_sorted[i-1]
            fm["prev"] = {"title": prv["title"], "url": f"/blog/{prv['slug']}"}
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


# ---------- Orchestration

def process_repo(name: str, url: str, ref: str | None) -> None:
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

    # Order strictly by filename path (natural sort) and build project blog link list
    posts_info.sort(key=lambda x: natural_key(x["rel_key"]))
    blog_posts = [{"title": p["title"], "url": f"/blog/{p['slug']}"} for p in posts_info]

    # prev/next pointers
    update_prev_next(posts_info)

    # Write/refresh the project card (kept feature)
    make_project_card(repo_dir, url, blog_posts=blog_posts if blog_posts else None)

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
