"""Jupyter notebook -> Markdown with raw-HTML outputs.

Cell visibility follows the Jupyter Book tag conventions:
  remove-cell, remove-input, remove-output   leave the cell / code / output out
  hide-input (or metadata.jupyter.source_hidden)   fold the code behind "Show code"
  show-input                                  never fold the code
  hide-cell, hide-output (or outputs_hidden)  treated like their remove-* versions
Code cells without a tag are folded when long (> FOLD_LINES) or only imports/setup,
so posts read as prose with the code a click away.
"""

from __future__ import annotations

import ast
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
FOLD_LINES = 15


def _tags(cell) -> set[str]:
    return {t.replace("_", "-") for t in cell.get("metadata", {}).get("tags", [])}


def _hidden(cell, what: str) -> bool:
    tags = _tags(cell)
    jupyter = cell.get("metadata", {}).get("jupyter", {})
    if what == "cell":
        return bool(tags & {"remove-cell", "hide-cell"})
    if what == "input":
        return "remove-input" in tags
    return bool(tags & {"remove-output", "hide-output"}) or bool(jupyter.get("outputs_hidden"))


def _statements(source: str) -> list[ast.stmt] | None:
    """Top-level statements, ignoring IPython magics; None if the code doesn't parse."""
    code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("%", "!")))
    try:
        return ast.parse(code).body
    except SyntaxError:
        return None


def _is_setup(stmts: list[ast.stmt] | None) -> bool:
    """Mostly imports (plus path/constant fiddling), no definitions."""
    if not stmts or any(isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for s in stmts):
        return False
    imports = sum(isinstance(s, (ast.Import, ast.ImportFrom)) for s in stmts)
    return imports > 0 and imports * 2 >= len(stmts)


def _fold_label(source: str, stmts: list[ast.stmt] | None) -> str:
    if _is_setup(stmts):
        return "Imports and setup"
    names = [s.name for s in stmts or [] if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    if names:
        shown = ", ".join(f"<code>{html.escape(n)}</code>" for n in names[:3])
        return f"Defines {shown}" + (f" and {len(names) - 3} more" if len(names) > 3 else "")
    first = next((ln.strip() for ln in source.splitlines() if ln.strip() and not ln.lstrip().startswith("#")), "")
    return f"<code>{html.escape(first if len(first) <= 60 else first[:57] + '…')}</code>"


def _code_block(cell, source: str, language: str) -> str:
    fence = f"```{language}\n{source}\n```"
    tags = _tags(cell)
    if "show-input" in tags:
        return fence
    stmts = _statements(source) if language == "python" else None
    lines = source.count("\n") + 1
    folded = (
        "hide-input" in tags
        or bool(cell.get("metadata", {}).get("jupyter", {}).get("source_hidden"))
        or lines > FOLD_LINES
        or (_is_setup(stmts) and lines > 3)  # a fold row is no shorter than a few lines of code
    )
    if not folded:
        return fence
    summary = (
        f'<summary><span class="nb-fold-label">{_fold_label(source, stmts)}</span>'
        f'<span class="nb-fold-meta">{lines} line{"s" if lines != 1 else ""}</span></summary>'
    )
    return f'<details class="nb-fold">\n{summary}\n\n{fence}\n\n</details>'


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
            if _hidden(cell, "input") or "hide-input" in _tags(cell) or not source:
                continue
            for att_name, bundle in (cell.get("attachments") or {}).items():
                mime, b64 = next(iter(bundle.items()))
                url = sink.add_bytes(f"attachment-{att_name}", base64.b64decode(b64))
                source = source.replace(f"attachment:{att_name}", url)
            blocks.append(rewrite_local_urls(source, path.parent, repo_dir, sink))

        elif cell.cell_type == "code":
            if source and not _hidden(cell, "input"):
                blocks.append(_code_block(cell, source, language))
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
