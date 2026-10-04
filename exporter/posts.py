from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .assets import MediaSink, rewrite_local_urls
from .git import first_and_last_commit
from .notebook import notebook_to_markdown
from .paths import GENERATED_DIR
from .util import natural_key, slugify, split_frontmatter, to_date, write_markdown

# File stems that say nothing about the post; the parent folder names it instead.
_GENERIC_STEMS = {"index", "readme", "notebook", "post", "main"}
_H1 = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


@dataclass
class SourcePost:
    path: Path
    slug: str


def discover(blog_dir: Path) -> list[SourcePost]:
    """Posts are blog/**/*.ipynb and blog/**/*.md, ordered by path (so number-prefix them)."""
    files = sorted(
        (p for p in blog_dir.rglob("*") if p.suffix in (".ipynb", ".md") and ".ipynb_checkpoints" not in p.parts),
        key=lambda p: natural_key(p.relative_to(blog_dir).as_posix()),
    )

    def short(p: Path) -> str:
        name = p.parent.name if p.stem.lower() in _GENERIC_STEMS and p.parent != blog_dir else p.stem
        return slugify(name)

    def full(p: Path) -> str:
        return slugify(p.relative_to(blog_dir).with_suffix("").as_posix())

    shorts = [short(p) for p in files]
    return [SourcePost(p, s if shorts.count(s) == 1 else full(p)) for p, s in zip(files, shorts)]


def _take_title(body: str) -> tuple[str | None, str]:
    """Pull the first H1 out of the body (the page renders the title itself)."""
    m = _H1.search(body)
    if not m:
        return None, body
    return m.group(1).strip(), body[: m.start()] + body[m.end() :]


def _excerpt(body: str, limit: int = 220) -> str | None:
    for para in re.split(r"\n\s*\n", body):
        p = para.strip()
        if not p or p[0] in "#<!|>-*`$[" or p.startswith("```"):
            continue
        p = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", p)
        p = re.sub(r"[*_`]", "", re.sub(r"\s+", " ", p))
        if len(p) > limit:
            p = p[:limit].rsplit(" ", 1)[0].rstrip(",.;:") + "…"
        return p
    return None


def export_post(
    post: SourcePost,
    *,
    order: int,
    project_id: str,
    repo: str,
    ref: str,
    repo_dir: Path,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    sink = MediaSink(f"posts/{project_id}/{post.slug}")

    if post.path.suffix == ".ipynb":
        meta, body = notebook_to_markdown(post.path, repo_dir, sink)
    else:
        meta, body = split_frontmatter(post.path.read_text(encoding="utf-8"))
        body = rewrite_local_urls(body, post.path.parent, repo_dir, sink)

    meta = {**meta, **overrides}
    h1, body = _take_title(body)
    first_commit, last_commit = first_and_last_commit(repo_dir, post.path)

    publish = to_date(meta.get("publishDate")) or first_commit or date.today()
    updated = to_date(meta.get("updatedDate")) or last_commit

    fm: dict[str, Any] = {
        "title": meta.get("title") or h1 or post.slug.replace("-", " ").capitalize(),
        "description": meta.get("description") or _excerpt(body),
        "publishDate": publish,
        "updatedDate": updated if updated and updated > publish else None,
        "draft": bool(meta.get("draft", False)),
        "project": project_id,
        "order": order,
        "tags": meta.get("tags") or [],
        "source": f"https://github.com/{repo}/blob/{ref}/{post.path.relative_to(repo_dir).as_posix()}",
    }
    fm = {k: v for k, v in fm.items() if v not in (None, [])}

    write_markdown(GENERATED_DIR / "posts" / project_id / post.slug / "index.md", fm, body)
    return fm
