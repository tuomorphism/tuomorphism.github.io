#!/usr/bin/env python3
"""
Robust notebook/markdown exporter for Astro collections.

- Posts -> src/content/post/<slug>/index.md
  frontmatter: title, publishDate, frontSlug, prev?, next?
- Projects -> src/content/projects/<slug>/index.md
  frontmatter: title, description, tier, image?, date?, links?, blog_posts?

Key features:
- Per-cell heading anchoring (no concat/split) with shared ID registry
- ATX + Setext headings → ATX with inline `{#id}`
- Code/math-safe mutations (fences + $…$/$$…$$)
- Markdown attachments (attachment:foo.png) extracted to assets
- Block HTML padded with blank lines
- URL rewrite + local asset copying with hashed names
- Deterministic names for nbconvert output blobs
- YAML date normalization to plain YYYY-MM-DD scalars
- CRLF normalization, nb validation
- Notebook/markdown assets mirrored to `public/blog/<slug>/assets/`
- Markdown/HTML URLs rewritten to `/blog/<slug>/assets/...`
- Projects get stable projectId and old project folders are removed when the slug changes.
"""

from __future__ import annotations

import sys
from typing import Dict, Any, List

import yaml

from .config import EXTERNAL, POST_OUT, PROJ_OUT, PUBLIC_BLOG, ROOT
from .git import clone_or_update
from .projects import make_project_card
from .utils import natural_key
from .posts import (
    export_markdown,
    export_notebook,
    update_prev_next,
)


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
    blog_posts = [
        {"title": p["title"], "url": f"/blog/{p['slug']}"}
        for p in posts_info
    ]

    update_prev_next(posts_info)
    make_project_card(
        repo_dir, url, blog_posts=blog_posts if blog_posts else None
    )


def main():
    EXTERNAL.mkdir(exist_ok=True)
    POST_OUT.mkdir(parents=True, exist_ok=True)
    PROJ_OUT.mkdir(parents=True, exist_ok=True)
    PUBLIC_BLOG.mkdir(parents=True, exist_ok=True)

    manifest_path = ROOT / "content-manifest.yml"
    if not manifest_path.exists():
        print(
            "ERROR: content-manifest.yml missing at repo root",
            file=sys.stderr,
        )
        sys.exit(1)

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    for r in manifest.get("repos", []):
        process_repo(r["name"], r["url"], r.get("ref"))


if __name__ == "__main__":
    main()
