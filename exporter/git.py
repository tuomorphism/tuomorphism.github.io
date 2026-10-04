from __future__ import annotations

import base64
import os
import subprocess
from datetime import date, datetime
from pathlib import Path


def _auth_args() -> list[str]:
    """Authenticate to GitHub with $GH_CONTENT_TOKEN (for private source repos) without storing it."""
    token = os.environ.get("GH_CONTENT_TOKEN")
    if not token:
        return []
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return ["-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {basic}"]


def _git(args: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *_auth_args(), *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


def sync(repo: str, ref: str, dest: Path) -> None:
    """Clone or fast-forward `repo` at `ref` into `dest`. Keeps history (for dates) but fetches blobs lazily."""
    url = f"https://github.com/{repo}.git"
    if (dest / ".git").exists():
        _git(["fetch", "--filter=blob:none", "origin", ref], cwd=dest)
        _git(["checkout", "--force", "--detach", "FETCH_HEAD"], cwd=dest)
        _git(["clean", "-fdx"], cwd=dest)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        _git(["clone", "--filter=blob:none", "--branch", ref, url, str(dest)])


def _commit_dates(repo_dir: Path, path: Path) -> list[date]:
    out = _git(["log", "--follow", "--format=%aI", "--", str(path.relative_to(repo_dir))], cwd=repo_dir)
    return [datetime.fromisoformat(line).date() for line in out.split()]


def first_and_last_commit(repo_dir: Path, path: Path) -> tuple[date | None, date | None]:
    dates = _commit_dates(repo_dir, path)
    return (dates[-1], dates[0]) if dates else (None, None)
