from __future__ import annotations

import pathlib
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional

from .assets import copy_project_image_for_repo
from .config import PROJ_OUT
from .utils import (
    normalize_frontmatter_dates,
    read_yaml,
    slugify,
    yaml_frontmatter_block,
    parse_frontmatter,
)


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


def _remove_old_project_dirs(project_id: str, current_slug: str) -> None:
    """
    Remove old project folders whose frontmatter has the same projectId
    but whose directory name (slug) no longer matches current_slug.
    """
    for child in PROJ_OUT.iterdir():
        if not child.is_dir():
            continue
        index_md = child / "index.md"
        if not index_md.exists():
            continue
        text = index_md.read_text(encoding="utf-8")
        fm, _ = parse_frontmatter(text)
        if not fm:
            continue
        if fm.get("projectId") == project_id and child.name != current_slug:
            print(f"- removing stale project folder {child.name}")
            shutil.rmtree(child, ignore_errors=True)


def make_project_card(
    repo_dir: pathlib.Path,
    repo_url: Optional[str],
    blog_posts: Optional[List[Dict[str, str]]] = None,
) -> None:
    meta = read_yaml(repo_dir / "project.yml")
    title = meta.get("title") or repo_dir.name.replace("-", " ").title()

    # Stable project id (used to find/remove old slugs)
    project_id = meta.get("projectId") or repo_dir.name

    slug = slugify(title)
    out_dir = PROJ_OUT / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    # Remove any previous folders for the same projectId but different slug
    _remove_old_project_dirs(project_id, slug)

    description = meta.get("description", "")
    tier = meta.get("tier", 2)

    image_override = meta.get("image") or meta.get("heroImage") or ""
    image = copy_project_image_for_repo(repo_dir, slug, override=image_override)

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
        "projectId": project_id,
    }
    if blog_posts:
        fm["blog_posts"] = blog_posts

    fm = normalize_frontmatter_dates(fm, keys=("date",))

    (out_dir / "index.md").write_text(
        yaml_frontmatter_block(fm), encoding="utf-8"
    )
    print(
        f"✓ project card {slug}"
        + (" with blog_posts" if blog_posts else "")
    )
