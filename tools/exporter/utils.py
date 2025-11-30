from __future__ import annotations

import hashlib
import pathlib
import re
from datetime import date, datetime
from typing import Any, Dict, Optional, Tuple

import yaml

from .config import (
    ROOT,
    SPACES_EOL,
    SLUG_RE,
)


def run(cmd, cwd=None):
    import subprocess
    print("+", " ".join(cmd), "[cwd=" + str(cwd or ROOT) + "]")
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def slugify(s: str) -> str:
    return re.sub(r"-{2,}", "-", SLUG_RE.sub("-", s.lower()).strip("-"))


def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s.lower())]


def read_yaml(path: pathlib.Path) -> Dict[str, Any]:
    if path.exists():
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {}


def content_hash(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _norm_text(s: str) -> str:
    return s.replace('\r\n', '\n').replace('\r', '\n').lstrip('\ufeff')


def _coerce_date_like(v):
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        s = v.strip().strip('"').strip("'")
        try:
            return date.fromisoformat(s)
        except ValueError:
            return v
    return v


def normalize_frontmatter_dates(
    fm: Dict[str, Any],
    keys=("publishDate", "date", "updateDate"),
) -> Dict[str, Any]:
    if not isinstance(fm, dict):
        return fm
    for k in keys:
        if k in fm:
            fm[k] = _coerce_date_like(fm[k])
    return fm


def yaml_frontmatter_block(data: Dict[str, Any]) -> str:
    def _fmt(v):
        if isinstance(v, datetime):
            return v.date().isoformat()
        if isinstance(v, date):
            return v.isoformat()
        return v

    data = normalize_frontmatter_dates({k: _fmt(v) for k, v in data.items()})
    dumped = yaml.safe_dump(
        data, sort_keys=False, allow_unicode=True
    ).rstrip()
    return f"---\n{dumped}\n---\n\n"


def parse_frontmatter(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    s = text.lstrip()
    if not s.startswith("---\n") and not s.startswith("---\r\n"):
        return None, text

    lines = text.splitlines(keepends=True)
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            fm_text = "".join(lines[1:i])
            body = "".join(lines[i + 1 :])
            fm = yaml.safe_load(fm_text) or {}
            return fm, body
    return None, text


def ensure_frontslug_in_frontmatter(md_path: pathlib.Path, slug: str) -> None:
    if not md_path.exists():
        return
    text = md_path.read_text(encoding="utf-8")
    fm, body = parse_frontmatter(text)
    fm = (fm or {})
    if fm.get("frontSlug") != slug:
        fm["frontSlug"] = slug
        md_path.write_text(
            yaml_frontmatter_block(fm) + body, encoding="utf-8"
        )


def normalize_markdown_light(md: str) -> str:
    md = SPACES_EOL.sub("", md)
    md = re.sub(r'\n{3,}', '\n\n', md)
    md = re.sub(r'([^\n])\n(#{1,6}\s)', r'\1\n\n\2', md)
    return md
