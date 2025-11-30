from __future__ import annotations

import hashlib
import pathlib
import shutil
from typing import Optional

from .config import (
    ASSET_DIR_NAME,
    ASSET_SOURCE_DIR_CANDIDATES,
    HTML_SRC_OR_HREF,
    MD_LINK_IMG,
    PUBLIC_BLOG,
    PUBLIC_PROJECTS,
)
from .utils import slugify


def ensure_dir(p: pathlib.Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def copy_project_image_for_repo(
    repo_dir: pathlib.Path,
    slug: str,
    override: str | None = None,
) -> str:
    """
    Returns the URL to be stored in frontmatter for the project image.

    Convention:
    - Default: assets/hero.* inside the project repo.
    - Optional: `override` can point to a custom relative path in the repo.
    - If nothing is found, returns "".
    """
    ensure_dir(PUBLIC_PROJECTS)

    if override:
        candidate = (repo_dir / override).resolve()
        if not candidate.exists():
            print(
                f"! override project image not found for {slug}: {override}"
            )
            return ""
    else:
        candidates = [
            repo_dir / "assets" / "hero.mp4",
            repo_dir / "assets" / "hero.webm",
            repo_dir / "assets" / "hero.gif",
            repo_dir / "assets" / "hero.png",
            repo_dir / "assets" / "hero.jpg",
            repo_dir / "assets" / "hero.jpeg",
            repo_dir / "assets" / "hero.webp",
        ]
        candidate = None
        for path in candidates:
            if path.exists():
                candidate = path.resolve()
                break
        if not candidate:
            print(f"- no hero media found in assets/ for {slug}, skipping")
            return ""

    if not candidate.exists():
        print(f"- candidate hero not found for {slug}, skipping")
        return ""

    data = candidate.read_bytes()
    h = hashlib.sha256(data).hexdigest()[:8]
    ext = candidate.suffix or ".png"
    fname = f"{slug}-{h}{ext}"
    out_path = PUBLIC_PROJECTS / fname
    out_path.write_bytes(data)

    return f"/projects/{fname}"


def is_relative_local(url: str) -> bool:
    import re

    if not url:
        return False
    if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', url):
        return False
    if url.startswith(("data:", "#", "/")):
        return False
    return True


def copy_asset_make_name(
    src: pathlib.Path, out_assets_dir: pathlib.Path
) -> str:
    data = src.read_bytes()
    h = hashlib.sha256(data).hexdigest()[:8]
    stem, ext = src.stem, src.suffix
    safe_stem = slugify(stem) or "asset"
    fname = f"{safe_stem}.{h}{ext}"
    ensure_dir(out_assets_dir)
    (out_assets_dir / fname).write_bytes(data)
    return f"{ASSET_DIR_NAME}/{fname}"


def resolve_asset_candidate(
    base_dir: pathlib.Path, url: str
) -> Optional[pathlib.Path]:
    cand = (base_dir / url).resolve()
    if cand.exists() and cand.is_file():
        return cand
    for adir in ASSET_SOURCE_DIR_CANDIDATES:
        cand2 = (base_dir / adir / url).resolve()
        if cand2.exists() and cand2.is_file():
            return cand2
    return None


def rewrite_urls_and_copy_assets(
    text: str,
    base_dir: pathlib.Path,
    out_assets_dir: pathlib.Path,
    url_prefix: str | None = None,
) -> str:
    """
    Rewrite markdown/HTML URLs that are local relative paths by:
    - looking up the source file near the notebook/markdown
    - copying it to out_assets_dir with a hashed name
    - returning either "assets/..." or "<url_prefix>/assets/..."
    """

    def _final_url(rel: str) -> str:
        if url_prefix:
            return f"{url_prefix.rstrip('/')}/{rel}"
        return rel

    def _md_repl(m):
        bang = m.group(1)
        alt = m.group("alt")
        url = m.group("url")
        if is_relative_local(url):
            src = resolve_asset_candidate(base_dir, url)
            if src:
                rel = copy_asset_make_name(src, out_assets_dir)
                return f"{bang}[{alt}]({_final_url(rel)})"
        return m.group(0)

    def _html_repl(m):
        attr = m.group("attr")
        url = m.group("url")
        if is_relative_local(url):
            src = resolve_asset_candidate(base_dir, url)
            if src:
                rel = copy_asset_make_name(src, out_assets_dir)
                return f'{attr}="{_final_url(rel)}"'
        return m.group(0)

    text = MD_LINK_IMG.sub(_md_repl, text)
    text = HTML_SRC_OR_HREF.sub(_html_repl, text)
    return text


def mirror_tree(src_dir: pathlib.Path, dst_dir: pathlib.Path) -> None:
    if not src_dir.exists():
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        return

    def walk_files(base: pathlib.Path) -> set[str]:
        out = set()
        for p in base.rglob("*"):
            if p.is_file():
                out.add(str(p.relative_to(base)))
        return out

    src_files = walk_files(src_dir)
    dst_files = walk_files(dst_dir) if dst_dir.exists() else set()

    dst_dir.mkdir(parents=True, exist_ok=True)

    for rel in src_files:
        s = src_dir / rel
        d = dst_dir / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        if (not d.exists()) or (
            hashlib.sha256(s.read_bytes()).hexdigest()
            != hashlib.sha256(d.read_bytes()).hexdigest()
        ):
            shutil.copy2(s, d)

    for rel in (dst_files - src_files):
        stale = dst_dir / rel
        try:
            stale.unlink()
        except IsADirectoryError:
            shutil.rmtree(stale, ignore_errors=True)


def sync_assets_to_public(slug: str, src_assets_dir: pathlib.Path) -> None:
    dest = PUBLIC_BLOG / slug / ASSET_DIR_NAME
    mirror_tree(src_assets_dir, dest)
