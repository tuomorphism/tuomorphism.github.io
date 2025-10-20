#!/usr/bin/env python3
"""
Robust notebook/markdown exporter for Astro collections.

- Posts -> src/content/post/<slug>/index.md
  frontmatter: title, publishDate, frontSlug, prev?, next?
- Projects -> src/content/projects/<slug>/index.md
  frontmatter: title, description, tier, image?, date?, links?, blog_posts?

Key hardening:
- Per-cell heading anchoring (no concat/split) with shared ID registry
- ATX + Setext headings → ATX with inline `{#id}` (no raw HTML anchors)
- Code/math-safe mutations (fences + $…$/$$…$$)
- Markdown attachments (attachment:foo.png) extracted to assets
- Block HTML padded with blank lines
- URL rewrite + local asset copying with hashed names
- Deterministic names for nbconvert output blobs
- YAML date normalization to plain YYYY-MM-DD scalars
- CRLF normalization, nb validation
"""

from __future__ import annotations
import sys, subprocess, pathlib, hashlib, base64, re
from datetime import datetime, date
from typing import Dict, Any, Optional, Tuple, List

import yaml
import nbformat
from nbformat.validator import validate
from nbconvert import MarkdownExporter

# ---------- Paths
ROOT = pathlib.Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "external"
POST_OUT = ROOT / "src" / "content" / "post"
PROJ_OUT = ROOT / "src" / "content" / "projects"
TEMPLATE_DIR = ROOT / "tools" / "templates"

# ---------- Config
ASSET_DIR_NAME = "assets"
ASSET_SOURCE_DIR_CANDIDATES = ("assets", "_assets")
MAX_TOC_DEPTH = 3

# ---------- Regex
_MD_LINK_IMG = re.compile(r'(!?)\[(?P<alt>[^\]]*)\]\((?P<url>[^)\s]+)(?:\s+"[^"]*")?\)')
_HTML_SRC_OR_HREF = re.compile(r'(?P<attr>\bsrc\b|\bhref\b)\s*=\s*([\'"])(?P<url>[^\'"]+)\2')
_MD_HEADING = re.compile(r'^(?P<hash>#{1,6})\s+(?P<text>.+?)\s*$', re.MULTILINE)
_SETEXT_RE = re.compile(r'^(?P<text>.+?)\n(?P<underline>=+|-+)\s*$', re.MULTILINE)
_ATTACHMENT_URL = re.compile(r'\battachment:(?P<name>[^)\s]+)')
_BLOCK_HTML = re.compile(r'^(<(?P<tag>(div|table|figure|video|iframe|details|summary|blockquote)\b)[\s\S]*?>[\s\S]*?</(?P=tag)>)$', re.MULTILINE)
_FENCE = re.compile(r"(^```.*?$)(.*?)(^```$)", re.MULTILINE | re.DOTALL)
_INLINE_MATH = re.compile(r'(?<!\\)\$(.+?)(?<!\\)\$')
_BLOCK_MATH  = re.compile(r'(^\$\$.*?^\$\$)', re.MULTILINE | re.DOTALL)
_SPACES_EOL = re.compile(r'[ \t]+$', re.MULTILINE)
_slug_re = re.compile(r"[^a-z0-9-]+")

# ---------- Utilities

def run(cmd, cwd=None):
    print("+", " ".join(cmd), "[cwd=" + str(cwd or ROOT) + "]")
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)

def slugify(s: str) -> str:
    return re.sub(r"-{2,}", "-", _slug_re.sub("-", s.lower()).strip("-"))

def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s.lower())]

def read_yaml(path: pathlib.Path) -> Dict[str, Any]:
    if path.exists():
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {}

def content_hash(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]

def _norm_text(s: str) -> str:
    return s.replace('\r\n', '\n').replace('\r', '\n').lstrip('\ufeff')

# --- Date normalization helpers (fixes quoted date strings)
def _coerce_date_like(v):
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        s = v.strip().strip('"').strip("'")
        try:
            return date.fromisoformat(s)
        except ValueError:
            return v
    return v

def normalize_frontmatter_dates(fm: Dict[str, Any], keys=("publishDate", "date", "updateDate")) -> Dict[str, Any]:
    if not isinstance(fm, dict):
        return fm
    for k in keys:
        if k in fm:
            fm[k] = _coerce_date_like(fm[k])
    return fm

def yaml_frontmatter_block(data: Dict[str, Any]) -> str:
    def _fmt(v):
        if isinstance(v, datetime):
            return v.date().isoformat()
        if isinstance(v, date):
            return v.isoformat()
        return v
    # Ensure any date-like values are normalized before dumping
    data = normalize_frontmatter_dates({k: _fmt(v) for k, v in data.items()})
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

# ---------- Asset helpers

def is_relative_local(url: str) -> bool:
    if not url:
        return False
    if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', url):
        return False
    if url.startswith(("data:", "#", "/")):
        return False
    return True

def ensure_dir(p: pathlib.Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def copy_asset_make_name(src: pathlib.Path, out_assets_dir: pathlib.Path) -> str:
    data = src.read_bytes()
    h = hashlib.sha256(data).hexdigest()[:8]
    stem, ext = src.stem, src.suffix
    safe_stem = slugify(stem) or "asset"
    fname = f"{safe_stem}.{h}{ext}"
    ensure_dir(out_assets_dir)
    (out_assets_dir / fname).write_bytes(data)
    return f"{ASSET_DIR_NAME}/{fname}"

def resolve_asset_candidate(base_dir: pathlib.Path, url: str) -> Optional[pathlib.Path]:
    cand = (base_dir / url).resolve()
    if cand.exists() and cand.is_file():
        return cand
    for adir in ASSET_SOURCE_DIR_CANDIDATES:
        cand2 = (base_dir / adir / url).resolve()
        if cand2.exists() and cand2.is_file():
            return cand2
    return None

def rewrite_urls_and_copy_assets(text: str, base_dir: pathlib.Path, out_assets_dir: pathlib.Path) -> str:
    def _md_repl(m: re.Match) -> str:
        bang = m.group(1)
        alt = m.group('alt')
        url = m.group('url')
        if is_relative_local(url):
            src = resolve_asset_candidate(base_dir, url)
            if src:
                rel = copy_asset_make_name(src, out_assets_dir)
                return f"{bang}[{alt}]({rel})"
        return m.group(0)

    def _html_repl(m: re.Match) -> str:
        attr = m.group('attr'); url = m.group('url')
        if is_relative_local(url):
            src = resolve_asset_candidate(base_dir, url)
            if src:
                rel = copy_asset_make_name(src, out_assets_dir)
                return f'{attr}="{rel}"'
        return m.group(0)

    text = _MD_LINK_IMG.sub(_md_repl, text)
    text = _HTML_SRC_OR_HREF.sub(_html_repl, text)
    return text

# ---------- Markdown cell processors (code/math safe)

def pad_block_html(md: str) -> str:
    lines, out, in_code = md.splitlines(), [], False
    for i, line in enumerate(lines):
        if line.strip().startswith(("```", "~~~")):
            in_code = not in_code
        if not in_code and _BLOCK_HTML.match(line):
            if out and out[-1] != "":
                out.append("")
            out.append(line)
            if i+1 < len(lines) and lines[i+1].strip() != "":
                out.append("")
            continue
        out.append(line)
    return "\n".join(out)

def map_noncode(md: str, fn):
    parts, last = [], 0
    for m in _FENCE.finditer(md):
        pre = md[last:m.start()]
        parts.append(fn(pre))
        parts.append(md[m.start():m.end()])
        last = m.end()
    parts.append(fn(md[last:]))
    return "".join(parts)

def map_noncode_nonmath(md: str, fn):
    def _strip_math(s):
        spans, tokens = [], []
        def _hold(regex, text):
            def repl(m):
                token = f"@@M{len(spans)}@@"
                spans.append(m.group(0))
                tokens.append(token)
                return token
            return regex.sub(repl, text)
        t = _hold(_BLOCK_MATH, s)
        t = _hold(_INLINE_MATH, t)
        t = fn(t)
        for token, span in zip(tokens, spans):
            t = t.replace(token, span, 1)
        return t
    return map_noncode(md, _strip_math)

def slugify_heading(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r'\s+', '-', s)
    s = re.sub(r'[^a-z0-9\-]', '', s)
    s = re.sub(r'-{2,}', '-', s).strip('-')
    return s or "section"

def add_ids_and_collect_toc_per_cell(md_text: str, used_ids: Dict[str, int], max_depth: int = 3) -> Tuple[str, List[Dict[str, Any]]]:
    """Add `{#id}` to headings (per cell) and collect ToC items."""
    toc: List[Dict[str, Any]] = []

    def unique_id(base: str) -> str:
        n = used_ids.get(base, 0)
        used_ids[base] = n + 1
        return base if n == 0 else f"{base}-{n}"

    # First: Setext → ATX with inline `{#id}`
    def _convert_setext(s: str) -> Tuple[str, List[Dict[str, Any]]]:
        local_toc: List[Dict[str, Any]] = []
        def _repl(m):
            level = 1 if m.group('underline').startswith('=') else 2
            text = m.group('text').strip()
            hid = unique_id(slugify_heading(text))
            if level <= max_depth:
                local_toc.append({"level": level, "text": text, "id": hid})
            return f"{'#'*level} {text} {{#{hid}}}"
        return _SETEXT_RE.sub(_repl, s), local_toc

    text, toc_setext = _convert_setext(md_text)
    toc.extend(toc_setext)

    # Then: ATX headings — append `{#id}` if missing
    def _inject_atx_ids(s: str) -> str:
        lines = s.splitlines()
        for i, line in enumerate(lines):
            m = _MD_HEADING.match(line)
            if not m:
                continue
            level = len(m.group('hash'))
            head_txt = m.group('text').strip()
            # Already has an id like '{#...}'?
            if re.search(r'\{\s*#[-a-z0-9]+?\s*\}\s*$', head_txt):
                # still collect for ToC
                if level <= max_depth:
                    id_match = re.search(r'\{\s*#([-a-z0-9]+)\s*\}\s*$', head_txt)
                    hid = id_match.group(1) if id_match else slugify_heading(re.sub(r'\s*\{#.*\}\s*$', '', head_txt))
                    text_only = re.sub(r'\s*\{#.*\}\s*$', '', head_txt)
                    toc.append({"level": level, "text": text_only, "id": hid})
                continue
            hid = unique_id(slugify_heading(head_txt))
            lines[i] = f"{'#'*level} {head_txt} {{#{hid}}}"
            if level <= max_depth:
                toc.append({"level": level, "text": head_txt, "id": hid})
        return "\n".join(lines)

    text = _inject_atx_ids(text)

    if md_text.endswith("\n") and not text.endswith("\n"):
        text += "\n"
    return text, toc

def normalize_markdown_light(md: str) -> str:
    md = _SPACES_EOL.sub('', md)
    md = re.sub(r'\n{3,}', '\n\n', md)
    md = re.sub(r'([^\n])\n(#{1,6}\s)', r'\1\n\n\2', md)  # blank line before heading
    return md

def extract_markdown_attachments(cell, out_assets_dir: pathlib.Path) -> Tuple[str, List[str]]:
    text = cell.get("source","")
    atts = cell.get("attachments") or {}
    used = []
    def _repl(m):
        name = m.group('name')
        blob = atts.get(name)
        if not blob:
            return m.group(0)
        mime, b64 = next(iter(blob.items()))
        ext = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/svg+xml": ".svg",
            "image/webp": ".webp",
        }.get(mime, ".bin")
        data = base64.b64decode(b64)
        h = hashlib.sha256(data).hexdigest()[:8]
        fname = f"att-{slugify(name)}.{h}{ext}"
        ensure_dir(out_assets_dir)
        (out_assets_dir / fname).write_bytes(data)
        used.append(fname)
        return f"{ASSET_DIR_NAME}/{fname}"
    text = _ATTACHMENT_URL.sub(_repl, text)
    return text, used

# ---------- Projects

def _merge_and_dedupe_links(links: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set(); deduped: List[Dict[str, str]] = []
    for li in links:
        if not isinstance(li, dict): continue
        url = li.get("url")
        if not url or url in seen: continue
        seen.add(url)
        deduped.append({"title": li.get("title") or "Link", "url": url})
    return deduped

def make_project_card(repo_dir: pathlib.Path, repo_url: Optional[str], blog_posts: Optional[List[Dict[str, str]]] = None) -> None:
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

    fm = normalize_frontmatter_dates(fm, keys=("date",))

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
    validate(nb)

    publish_date = git_last_commit_date(repo_dir, ipynb) or datetime.now().date()

    # Per-markdown-cell processing with shared heading ID registry
    base_dir = ipynb.parent
    toc_items: List[Dict[str, Any]] = []
    used_ids: Dict[str, int] = {}

    for cell in nb.cells:
        if cell.get("cell_type") != "markdown":
            continue
        raw = _norm_text(cell.get("source", ""))

        # attachments
        raw, _ = extract_markdown_attachments({"source": raw, "attachments": cell.get("attachments") or {}}, out_assets_dir)

        # rewrite/copy assets
        raw = rewrite_urls_and_copy_assets(raw, base_dir, out_assets_dir)

        # pad block HTML & add heading IDs (code/math safe)
        raw = map_noncode(raw, pad_block_html)

        def _ids(s):
            s2, toc_cell = add_ids_and_collect_toc_per_cell(s, used_ids, max_depth=MAX_TOC_DEPTH)
            if toc_cell: toc_items.extend(toc_cell)
            return s2

        raw = map_noncode_nonmath(raw, _ids)

        # light normalization (safe)
        raw = map_noncode_nonmath(raw, normalize_markdown_light)

        cell["source"] = raw

    # Title: prefer nb.metadata.title then first H1
    first_h1 = None
    h1_re = re.compile(r'^\s*#\s+(.+?)\s*(?:\{\s*#[-a-z0-9]+\s*\})?\s*$', re.MULTILINE)
    for cell in nb.cells:
        if cell.get("cell_type") != "markdown": continue
        m = h1_re.search(cell.get("source",""))
        if m:
            first_h1 = m.group(1).strip(); break
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
    resources = {"metadata": {"title": title, "publishDate": publish_date, "frontSlug": slug, "toc_items": toc_items}}

    body, res = exporter.from_notebook_node(nb, resources=resources)

    # Deterministic names for output blobs + patch references
    outputs = res.get("outputs") or {}
    for name, data in list(outputs.items()):
        p = pathlib.Path(name)
        h = hashlib.sha256(data).hexdigest()[:8]
        new_name = f"{slugify(p.stem)}.{h}{p.suffix}"
        (out_dir / new_name).write_bytes(data)
        if new_name != name:
            body = body.replace(name, new_name)

    # Normalize the frontmatter emitted by the Jinja template (fix quoted dates)
    fm_rendered, body_md = parse_frontmatter(body)
    if fm_rendered is None:
        fm_rendered = {"title": title, "publishDate": publish_date, "frontSlug": slug, "toc_items": toc_items}
    else:
        # Ensure required keys exist
        fm_rendered.setdefault("title", title)
        fm_rendered.setdefault("publishDate", publish_date)
        fm_rendered.setdefault("frontSlug", slug)
        fm_rendered.setdefault("toc_items", toc_items)

    fm_rendered = normalize_frontmatter_dates(fm_rendered)
    body = yaml_frontmatter_block(fm_rendered) + body_md

    (out_dir / "index.md").write_text(body, encoding="utf-8")
    ensure_frontslug_in_frontmatter(out_dir / "index.md", slug)

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
    text = _norm_text(md.read_text(encoding="utf-8"))
    fm, body = parse_frontmatter(text)

    base_dir = md.parent

    used_ids: Dict[str, int] = {}
    toc_items: List[Dict[str, Any]] = []

    # rewrite/copy assets + html padding
    body = rewrite_urls_and_copy_assets(body, base_dir, out_assets_dir) if fm is not None else rewrite_urls_and_copy_assets(text, base_dir, out_assets_dir)
    body = map_noncode(body, pad_block_html)

    # heading IDs (setext + atx)
    def _ids(s):
        s2, toc_cell = add_ids_and_collect_toc_per_cell(s, used_ids, max_depth=MAX_TOC_DEPTH)
        if toc_cell: toc_items.extend(toc_cell)
        return s2
    body = map_noncode_nonmath(body, _ids)
    body = map_noncode_nonmath(body, normalize_markdown_light)

    title = (fm.get("title") if fm else md.stem.title())
    publishDate = (fm.get("publishDate") if fm else git_last_commit_date(repo_dir, md)) or datetime.now().date()

    if marker.exists():
        ensure_frontslug_in_frontmatter(out_dir / "index.md", slug)
        print(f"= {slug} unchanged, skip")
        return {"title": title, "slug": slug, "publishDate": publishDate, "rel_key": rel_key}

    if not fm:
        fm = {"title": title, "publishDate": publishDate, "frontSlug": slug}
    else:
        fm.setdefault("title", title)
        fm.setdefault("publishDate", publishDate)
        fm["frontSlug"] = slug

    fm = normalize_frontmatter_dates(fm)

    (out_dir / "index.md").write_text(yaml_frontmatter_block(fm) + body, encoding="utf-8")
    ensure_frontslug_in_frontmatter(out_dir / "index.md", slug)

    for old in out_dir.glob(".hash-*"):
        old.unlink()
    marker.touch()
    print(f"✓ exported markdown {slug}")

    return {"title": title, "slug": slug, "publishDate": publishDate, "rel_key": rel_key}

def update_prev_next(posts_sorted: List[Dict[str, Any]]) -> None:
    for i, p in enumerate(posts_sorted):
        post_md = POST_OUT / p["slug"] / "index.md"
        if not post_md.exists():
            continue
        text = post_md.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(text)
        fm = (fm or {})
        fm["frontSlug"] = p["slug"]  # keep required key stable

        if i > 0:
            prv = posts_sorted[i-1]
            fm["prev"] = {"title": prv["title"], "url": f"/blog/{prv['slug']}"}
        else:
            fm.pop("prev", None)

        if i < len(posts_sorted) - 1:
            nxt = posts_sorted[i+1]
            fm["next"] = {"title": nxt["title"], "url": f"/blog/{nxt['slug']}"}
        else:
            fm.pop("next", None)

        fm = normalize_frontmatter_dates(fm)  # keep any dates clean if present
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

    posts_info.sort(key=lambda x: natural_key(x["rel_key"]))
    blog_posts = [{"title": p["title"], "url": f"/blog/{p['slug']}"} for p in posts_info]

    update_prev_next(posts_info)
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
