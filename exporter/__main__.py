from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

from .git import sync
from .paths import CACHE_DIR, GENERATED_DIR, MEDIA_DIR, SOURCES_FILE
from .posts import discover, export_post
from .project import export_project
from .util import read_yaml


def _clean(project_id: str | None = None) -> None:
    """Remove previous output (all of it, or one project's) so nothing stale survives."""
    if project_id is None:
        shutil.rmtree(GENERATED_DIR, ignore_errors=True)
        shutil.rmtree(MEDIA_DIR, ignore_errors=True)
        return
    (GENERATED_DIR / "projects" / f"{project_id}.yaml").unlink(missing_ok=True)
    for d in (GENERATED_DIR / "posts" / project_id, MEDIA_DIR / "posts" / project_id, MEDIA_DIR / "projects" / project_id):
        shutil.rmtree(d, ignore_errors=True)


def export_source(src: dict) -> None:
    project_id, repo, ref = src["id"], src["repo"], src.get("ref", "main")
    print(f"→ {project_id} ({repo}@{ref})")
    repo_dir = CACHE_DIR / project_id
    sync(repo, ref, repo_dir)

    export_project(project_id, repo, repo_dir, src.get("project") or {})

    blog_dir = repo_dir / "blog"
    posts = discover(blog_dir) if blog_dir.is_dir() else []
    post_overrides = src.get("posts") or {}
    for unknown in set(post_overrides) - {p.slug for p in posts}:
        print(f"  ! override for unknown post '{unknown}'")
    for order, post in enumerate(posts, start=1):
        fm = export_post(
            post,
            order=order,
            project_id=project_id,
            repo=repo,
            ref=ref,
            repo_dir=repo_dir,
            overrides=post_overrides.get(post.slug) or {},
        )
        print(f"  ✓ /blog/{project_id}/{post.slug}  {fm['title']!r}")


def main() -> int:
    parser = argparse.ArgumentParser(prog="python3 -m exporter", description=__doc__)
    parser.add_argument("--only", nargs="+", metavar="ID", help="export only these source ids")
    parser.add_argument("--keep-going", action="store_true", help="skip sources that fail instead of aborting")
    args = parser.parse_args()

    sources = read_yaml(SOURCES_FILE).get("sources") or []
    if args.only:
        unknown = set(args.only) - {s["id"] for s in sources}
        if unknown:
            parser.error(f"unknown source id(s): {', '.join(sorted(unknown))}")
        sources = [s for s in sources if s["id"] in args.only]
        for s in sources:
            _clean(s["id"])
    else:
        _clean()

    failed = []
    for src in sources:
        try:
            export_source(src)
        except (subprocess.CalledProcessError, OSError, ValueError) as e:
            detail = e.stderr.strip() if isinstance(e, subprocess.CalledProcessError) and e.stderr else e
            print(f"  ✗ {src['id']}: {detail}", file=sys.stderr)
            if not args.keep_going:
                return 1
            _clean(src["id"])
            failed.append(src["id"])

    if failed:
        print(f"Skipped failed sources: {', '.join(failed)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
