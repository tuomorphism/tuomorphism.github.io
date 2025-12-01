from __future__ import annotations

import subprocess
from datetime import datetime, date
from pathlib import Path
from typing import Optional
from .config import EXTERNAL
from .utils import run


def clone_or_update(name: str, url: str, ref: str | None) -> Path:
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

def _run_git_dates(repo_dir: Path, args: list[str]) -> list[date]:
    proc = subprocess.run(
        ["git"] + args,
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    dates: list[date] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # ISO 8601, e.g. 2025-03-01T10:23:45+00:00
        dt = datetime.fromisoformat(line)
        dates.append(dt.date())
    return dates


def git_last_commit_date(repo_dir: Path, path: Path) -> Optional[date]:
    rel = path.relative_to(repo_dir)
    dates = _run_git_dates(
        repo_dir,
        ["log", "--follow", "-1", "--format=%aI", "--", str(rel)],
    )
    return dates[0] if dates else None


def git_first_commit_date(repo_dir: Path, path: Path) -> Optional[date]:
    """
    First commit touching this path (oldest dev time).
    """
    rel = path.relative_to(repo_dir)
    dates = _run_git_dates(
        repo_dir,
        ["log", "--follow", "--reverse", "--format=%aI", "--", str(rel)],
    )
    return dates[0] if dates else None
