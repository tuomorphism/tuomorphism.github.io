"""Jupyter notebook -> Markdown with raw-HTML outputs.

Cell visibility follows the usual Jupyter/Jupyter Book conventions:
  tags remove-cell / remove-input / remove-output (and hide-* variants),
  or metadata.jupyter.source_hidden / outputs_hidden.
"""

from __future__ import annotations

import base64
import html
import re
from pathlib import Path
from typing import Any

import nbformat

from .assets import MediaSink, rewrite_local_urls

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_STYLE = re.compile(r"<style[\s\S]*?</style>", re.IGNORECASE)
_IMAGE_TYPES = {"image/svg+xml": ".svg", "image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif"}


def _tags(cell) -> set[str]:
    return {t.replace("_", "-") for t in cell.get("metadata", {}).get("tags", [])}


def _hidden(cell, what: str) -> bool:
    tags = _tags(cell)
    jupyter = cell.get("metadata", {}).get("jupyter", {})
    if what == "cell":
        return bool(tags & {"remove-cell", "hide-cell"})
    if what == "input":
        return bool(tags & {"remove-input", "hide-input"}) or bool(jupyter.get("source_hidden"))
    return bool(tags & {"remove-output", "hide-output"}) or bool(jupyter.get("outputs_hidden"))


def _pre(text: str) -> str:
    return f'<pre class="nb-text"><code>{html.escape(_ANSI.sub("", text).rstrip())}</code></pre>'


def _render_output(out: dict[str, Any], sink: MediaSink, name: str) -> str | None:
    kind = out.get("output_type")
    if kind == "stream":
        return _pre(out.get("text", ""))
    if kind == "error":
        return _pre("\n".join([f"{out.get('ename')}: {out.get('evalue')}", *out.get("traceback", [])]))

    data = out.get("data", {})
    for mime, ext in _IMAGE_TYPES.items():
        if mime in data:
            raw = data[mime]
            blob = raw.encode() if mime == "image/svg+xml" else base64.b64decode(raw)
            return f'<img src="{sink.add_bytes(name + ext, blob)}" alt="" loading="lazy">'
    if "text/html" in data:
        return _STYLE.sub("", data["text/html"]).strip()
    if "text/markdown" in data:
        return data["text/markdown"]
    if "text/latex" in data:
        return data["text/latex"]
    if "text/plain" in data:
        return _pre(data["text/plain"])
    return None


def notebook_to_markdown(path: Path, repo_dir: Path, sink: MediaSink) -> tuple[dict[str, Any], str]:
    """Returns (notebook metadata, markdown body)."""
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    language = nb.metadata.get("language_info", {}).get("name", "python")

    blocks: list[str] = []
    for i, cell in enumerate(nb.cells):
        if _hidden(cell, "cell"):
            continue
        source = cell.source.replace("\r\n", "\n").replace("\u200b", "").strip()

        if cell.cell_type == "markdown":
            if _hidden(cell, "input") or not source:
                continue
            for att_name, bundle in (cell.get("attachments") or {}).items():
                mime, b64 = next(iter(bundle.items()))
                url = sink.add_bytes(f"attachment-{att_name}", base64.b64decode(b64))
                source = source.replace(f"attachment:{att_name}", url)
            blocks.append(rewrite_local_urls(source, path.parent, repo_dir, sink))

        elif cell.cell_type == "code":
            if source and not _hidden(cell, "input"):
                blocks.append(f"```{language}\n{source}\n```")
            if _hidden(cell, "output"):
                continue
            rendered = [
                r
                for j, out in enumerate(cell.get("outputs", []))
                if (r := _render_output(out, sink, f"output-{i}-{j}"))
            ]
            if rendered:
                # Blank lines around the wrapper keep Markdown inside it (e.g. text/markdown) parseable.
                blocks.append('<div class="nb-output">\n\n' + "\n\n".join(rendered) + "\n\n</div>")

        elif cell.cell_type == "raw":
            fmt = cell.get("metadata", {}).get("format", "")
            if source and fmt in ("text/markdown", "text/html"):
                blocks.append(source)

    return dict(nb.metadata), "\n\n".join(blocks)
