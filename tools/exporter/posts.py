from __future__ import annotations

import hashlib
import pathlib
import re
from datetime import datetime
from typing import Any, Dict, List

import nbformat
from nbconvert import MarkdownExporter
from nbformat.validator import validate

from .assets import (
    rewrite_urls_and_copy_assets,
    sync_assets_to_public,
)
from .config import (
    ASSET_DIR_NAME,
    MAX_TOC_DEPTH,
    POST_OUT,
    TEMPLATE_DIR,
)
from .git import git_last_commit_date
from .markdown_processing import (
    add_ids_and_collect_toc_per_cell,
    extract_markdown_attachments,
    map_noncode,
    map_noncode_nonmath,
    pad_block_html,
)
from .utils import (
    content_hash,
    ensure_frontslug_in_frontmatter,
    normalize_markdown_light,
    normalize_frontmatter_dates,
    parse_frontmatter,
    slugify,
    yaml_frontmatter_block,
    _norm_text,
)
from .visibility import filter_and_apply_visibility


def export_notebook(
    ipynb: pathlib.Path,
    repo_name: str,
    rel_key: str,
    repo_dir: pathlib.Path,
) -> Dict[str, Any]:
    slug = slugify(f"{repo_name}-{rel_key}")
    out_dir = POST_OUT / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_assets_dir = out_dir / ASSET_DIR_NAME

    marker = out_dir / f".hash-{content_hash(ipynb)}"

    nb = nbformat.read(str(ipynb), as_version=4)
    validate(nb)

    filter_and_apply_visibility(nb)

    publish_date = git_last_commit_date(repo_dir, ipynb) or datetime.now().date()

    base_dir = ipynb.parent
    toc_items: List[Dict[str, Any]] = []
    used_ids: Dict[str, int] = {}

    for cell in nb.cells:
        if cell.get("cell_type") != "markdown":
            continue
        raw = _norm_text(cell.get("source", ""))

        # attachments
        raw, _ = extract_markdown_attachments(
            {"source": raw, "attachments": cell.get("attachments") or {}},
            out_assets_dir,
        )

        # rewrite/copy assets — now with absolute `/blog/<slug>/...`
        raw = rewrite_urls_and_copy_assets(
            raw,
            base_dir,
            out_assets_dir,
            url_prefix=f"/blog/{slug}",
        )

        # pad block HTML & add heading IDs
        raw = map_noncode(raw, pad_block_html)

        def _ids(s):
            s2, toc_cell = add_ids_and_collect_toc_per_cell(
                s, used_ids, max_depth=MAX_TOC_DEPTH
            )
            if toc_cell:
                toc_items.extend(toc_cell)
            return s2

        raw = map_noncode_nonmath(raw, _ids)
        raw = map_noncode_nonmath(raw, normalize_markdown_light)

        cell["source"] = raw

    # Title detection
    first_h1 = None
    h1_re = re.compile(
        r'^\s*#\s+(.+?)\s*(?:\{\s*#[-a-z0-9]+\s*\})?\s*$',
        re.MULTILINE,
    )
    for cell in nb.cells:
        if cell.get("cell_type") != "markdown":
            continue
        m = h1_re.search(cell.get("source", ""))
        if m:
            first_h1 = m.group(1).strip()
            break
    title = (
        nb.metadata.get("title")
        or first_h1
        or f"{repo_name}: {ipynb.stem.replace('-', ' ').title()}"
    )

    if marker.exists():
        ensure_frontslug_in_frontmatter(out_dir / "index.md", slug)
        # still mirror assets; even if content unchanged, assets might have changed
        sync_assets_to_public(slug, out_assets_dir)
        print(f"= {slug} unchanged, skip")
        return {
            "title": title,
            "slug": slug,
            "publishDate": publish_date,
            "rel_key": rel_key,
        }

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

    # output blobs
    outputs = res.get("outputs") or {}
    for name, data in list(outputs.items()):
        p = pathlib.Path(name)
        h = hashlib.sha256(data).hexdigest()[:8]
        new_name = f"{slugify(p.stem)}.{h}{p.suffix}"
        (out_dir / new_name).write_bytes(data)
        if new_name != name:
            body = body.replace(name, new_name)

    fm_rendered, body_md = parse_frontmatter(body)
    if fm_rendered is None:
        fm_rendered = {
            "title": title,
            "publishDate": publish_date,
            "frontSlug": slug,
            "toc_items": toc_items,
        }
    else:
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

    # mirror assets so /blog/<slug>/assets/... exists
    sync_assets_to_public(slug, out_assets_dir)

    print(f"✓ exported notebook {slug}")
    return {
        "title": title,
        "slug": slug,
        "publishDate": publish_date,
        "rel_key": rel_key,
    }


def export_markdown(
    md: pathlib.Path,
    repo_name: str,
    rel_key: str,
    repo_dir: pathlib.Path,
) -> Dict[str, Any]:
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

    if fm is not None:
        body = rewrite_urls_and_copy_assets(
            body,
            base_dir,
            out_assets_dir,
            url_prefix=f"/blog/{slug}",
        )
    else:
        body = rewrite_urls_and_copy_assets(
            text,
            base_dir,
            out_assets_dir,
            url_prefix=f"/blog/{slug}",
        )

    body = map_noncode(body, pad_block_html)

    def _ids(s):
        s2, toc_cell = add_ids_and_collect_toc_per_cell(
            s, used_ids, max_depth=MAX_TOC_DEPTH
        )
        if toc_cell:
            toc_items.extend(toc_cell)
        return s2

    body = map_noncode_nonmath(body, _ids)
    body = map_noncode_nonmath(body, normalize_markdown_light)

    title = (fm.get("title") if fm else md.stem.title())
    publishDate = (
        (fm.get("publishDate") if fm else git_last_commit_date(repo_dir, md))
        or datetime.now().date()
    )

    if marker.exists():
        ensure_frontslug_in_frontmatter(out_dir / "index.md", slug)
        # still mirror assets
        sync_assets_to_public(slug, out_assets_dir)
        print(f"= {slug} unchanged, skip")
        return {
            "title": title,
            "slug": slug,
            "publishDate": publishDate,
            "rel_key": rel_key,
        }

    if not fm:
        fm = {"title": title, "publishDate": publishDate, "frontSlug": slug}
    else:
        fm.setdefault("title", title)
        fm.setdefault("publishDate", publishDate)
        fm["frontSlug"] = slug

    fm = normalize_frontmatter_dates(fm)

    (out_dir / "index.md").write_text(
        yaml_frontmatter_block(fm) + body, encoding="utf-8"
    )
    ensure_frontslug_in_frontmatter(out_dir / "index.md", slug)

    for old in out_dir.glob(".hash-*"):
        old.unlink()
    marker.touch()

    # mirror assets for markdown posts too
    sync_assets_to_public(slug, out_assets_dir)

    print(f"✓ exported markdown {slug}")

    return {
        "title": title,
        "slug": slug,
        "publishDate": publishDate,
        "rel_key": rel_key,
    }


def update_prev_next(posts_sorted: List[Dict[str, Any]]) -> None:
    for i, p in enumerate(posts_sorted):
        post_md = POST_OUT / p["slug"] / "index.md"
        if not post_md.exists():
            continue
        text = post_md.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(text)
        fm = (fm or {})
        fm["frontSlug"] = p["slug"]

        if i > 0:
            prv = posts_sorted[i - 1]
            fm["prev"] = {
                "title": prv["title"],
                "url": f"/blog/{prv['slug']}",
            }
        else:
            fm.pop("prev", None)

        if i < len(posts_sorted) - 1:
            nxt = posts_sorted[i + 1]
            fm["next"] = {
                "title": nxt["title"],
                "url": f"/blog/{nxt['slug']}",
            }
        else:
            fm.pop("next", None)

        fm = normalize_frontmatter_dates(fm)
        post_md.write_text(
            yaml_frontmatter_block(fm) + body, encoding="utf-8"
        )
    if posts_sorted:
        print(f"✓ updated prev/next for {len(posts_sorted)} posts")
