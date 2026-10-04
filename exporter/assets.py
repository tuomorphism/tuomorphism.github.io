from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .paths import MEDIA_DIR, MEDIA_URL
from .util import slugify

_MD_LINK = re.compile(r"(?P<pre>!?\[[^\]]*\]\()(?P<url>[^)\s]+)(?P<post>(?:\s+\"[^\"]*\")?\))")
_HTML_ATTR = re.compile(r"(?P<pre>\b(?:src|href|poster)\s*=\s*)(?P<q>['\"])(?P<url>[^'\"]+)(?P=q)")
_FENCE = re.compile(r"^(```|~~~).*?^\1\s*$", re.MULTILINE | re.DOTALL)


class MediaSink:
    """Copies files into public/media/<namespace>/ under content-hashed names and returns their URLs."""

    def __init__(self, namespace: str):
        self.dir = MEDIA_DIR / namespace
        self.url = f"{MEDIA_URL}/{namespace}"

    def add_bytes(self, name: str, data: bytes) -> str:
        p = Path(name)
        digest = hashlib.sha256(data).hexdigest()[:8]
        fname = f"{slugify(p.stem) or 'asset'}.{digest}{p.suffix.lower()}"
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / fname).write_bytes(data)
        return f"{self.url}/{fname}"

    def add_file(self, path: Path) -> str:
        return self.add_bytes(path.name, path.read_bytes())


def _is_relative(url: str) -> bool:
    return not re.match(r"^([a-z][a-z0-9+.-]*:|/|#)", url, re.IGNORECASE)


def _resolve(url: str, base_dir: Path, repo_dir: Path) -> Path | None:
    rel = url.split("#")[0].split("?")[0]
    for candidate in (base_dir / rel, base_dir / "assets" / rel):
        p = candidate.resolve()
        if p.is_file() and p.is_relative_to(repo_dir.resolve()):
            return p
    return None


def rewrite_local_urls(text: str, base_dir: Path, repo_dir: Path, sink: MediaSink) -> str:
    """Point relative links/images (markdown and HTML) at copies in the media sink. Code fences are left alone."""

    def repl(m: re.Match) -> str:
        url = m.group("url")
        if not _is_relative(url):
            return m.group(0)
        src = _resolve(url, base_dir, repo_dir)
        if src is None:
            print(f"  ! unresolved local reference: {url}")
            return m.group(0)
        return m.group(0).replace(url, sink.add_file(src), 1)

    def rewrite(chunk: str) -> str:
        return _HTML_ATTR.sub(repl, _MD_LINK.sub(repl, chunk))

    out, last = [], 0
    for fence in _FENCE.finditer(text):
        out += [rewrite(text[last : fence.start()]), fence.group(0)]
        last = fence.end()
    out.append(rewrite(text[last:]))
    return "".join(out)
