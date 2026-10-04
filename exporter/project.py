from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from .assets import MediaSink
from .paths import GENERATED_DIR
from .util import read_yaml, to_date, write_yaml

_COVER_EXTS = (".mp4", ".webm", ".gif", ".png", ".jpg", ".jpeg", ".webp")


def _find_cover(repo_dir: Path, override: str | None) -> Path | None:
    if override:
        p = (repo_dir / override).resolve()
        if p.is_file() and p.is_relative_to(repo_dir.resolve()):
            return p
        print(f"  ! cover not found: {override}")
        return None
    return next((p for ext in _COVER_EXTS if (p := repo_dir / "assets" / f"hero{ext}").is_file()), None)


def export_project(project_id: str, repo: str, repo_dir: Path, overrides: dict[str, Any]) -> dict[str, Any]:
    """Writes generated/projects/<id>.yaml from the repo's project.yml (+ overrides from sources.yml)."""
    meta = {**read_yaml(repo_dir / "project.yml"), **overrides}

    links = [{"label": "Repo", "url": f"https://github.com/{repo}"}]
    for li in meta.get("links") or []:
        url = li.get("url")
        if url and all(url != existing["url"] for existing in links):
            links.append({"label": li.get("label") or li.get("title") or "Link", "url": url})
    if meta.get("demo_url"):
        links.append({"label": "Demo", "url": meta["demo_url"]})

    cover = _find_cover(repo_dir, meta.get("cover") or meta.get("image") or None)

    data: dict[str, Any] = {
        "title": meta.get("title") or project_id.replace("-", " ").capitalize(),
        "summary": meta.get("summary") or meta.get("description") or "",
        "featured": bool(meta.get("featured", meta.get("tier") == 1)),
        "date": to_date(meta.get("date")) or date.today(),
        "cover": MediaSink(f"projects/{project_id}").add_file(cover) if cover else None,
        "links": links,
        "tags": meta.get("tags") or [],
    }
    data = {k: v for k, v in data.items() if v not in (None, [])}
    write_yaml(GENERATED_DIR / "projects" / f"{project_id}.yaml", data)
    return data
