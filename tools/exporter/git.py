from __future__ import annotations

import pathlib
import subprocess
from datetime import date, datetime
from typing import Optional

from .config import EXTERNAL
from .utils import run


def clone_or_update(name: str, url: str, ref: str | None) -> pathlib.Path:
    dest = EXTERNAL / name
    if dest.exists():
        run(["git", "fetch", "origin", "--prune"], cwd=dest)
        run(["git", "checkout", ref or "main"], cwd=dest)
        try:
            run(
                ["git", "reset", "--hard", f"origin/{ref or 'main'}"],
                cwd=dest,
            )
        except subprocess.CalledProcessError:
            # Branch may not exist, keep last checked-out state
            pass
    else:
        run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                ref or "main",
                url,
                str(dest),
            ]
        )
    return dest


def git_last_commit_date(
    repo_root: pathlib.Path, path: pathlib.Path
) -> Optional[date]:
    try:
        out = subprocess.check_output(
            ["git", "log", "-1", "--format=%cI", "--", str(path)],
            cwd=str(repo_root),
        ).decode().strip()
        if out:
            return datetime.fromisoformat(out).date()
    except subprocess.CalledProcessError:
        return None
    return None
