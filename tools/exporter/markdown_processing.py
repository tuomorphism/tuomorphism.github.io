from __future__ import annotations

import base64
import hashlib
from typing import Any, Dict, List, Tuple

from .assets import ensure_dir
from .config import (
    ASSET_DIR_NAME,
    ATTACHMENT_URL,
    BLOCK_HTML,
    BLOCK_MATH,
    FENCE,
    INLINE_MATH,
    MD_HEADING,
    SETEXT_RE,
)
from .utils import (
    slugify,
)


def pad_block_html(md: str) -> str:
    lines, out, in_code = md.splitlines(), [], False
    for i, line in enumerate(lines):
        if line.strip().startswith(("```", "~~~")):
            in_code = not in_code
        if (not in_code) and BLOCK_HTML.match(line):
            if out and out[-1] != "":
                out.append("")
            out.append(line)
            if i + 1 < len(lines) and lines[i + 1].strip() != "":
                out.append("")
            continue
        out.append(line)
    return "\n".join(out)


def map_noncode(md: str, fn):
    parts, last = [], 0
    for m in FENCE.finditer(md):
        pre = md[last : m.start()]
        parts.append(fn(pre))
        parts.append(md[m.start() : m.end()])
        last = m.end()
    parts.append(fn(md[last:]))
    return "".join(parts)


def map_noncode_nonmath(md: str, fn):
    def _strip_math(s):
        spans, tokens = [], []

        def _hold(regex, text):
            def repl(m):
                token = f"@@M{len(spans)}@@"
                spans.append(m.group(0))
                tokens.append(token)
                return token

            return regex.sub(repl, text)

        t = _hold(BLOCK_MATH, s)
        t = _hold(INLINE_MATH, t)
        t = fn(t)
        for token, span in zip(tokens, spans):
            t = t.replace(token, span, 1)
        return t

    return map_noncode(md, _strip_math)


def slugify_heading(text: str) -> str:
    import re

    s = text.strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9\-]", "", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "section"


def add_ids_and_collect_toc_per_cell(
    md_text: str, used_ids: Dict[str, int], max_depth: int = 3
) -> Tuple[str, List[Dict[str, Any]]]:
    """Add `{#id}` to headings (per cell) and collect ToC items."""
    toc: List[Dict[str, Any]] = []

    def unique_id(base: str) -> str:
        n = used_ids.get(base, 0)
        used_ids[base] = n + 1
        return base if n == 0 else f"{base}-{n}"

    # 1) Setext → ATX with inline {#id}
    def _convert_setext(s: str):
        local_toc: List[Dict[str, Any]] = []

        def _repl(m):
            level = 1 if m.group("underline").startswith("=") else 2
            text = m.group("text").strip()
            hid = unique_id(slugify_heading(text))
            if level <= max_depth:
                local_toc.append({"level": level, "text": text, "id": hid})
            return f"{'#' * level} {text} {{#{hid}}}"

        return SETEXT_RE.sub(_repl, s), local_toc

    text, toc_setext = _convert_setext(md_text)
    toc.extend(toc_setext)

    # 2) ATX headings → ensure {#id} exists
    def _inject_atx_ids(s: str) -> str:
        import re

        lines = s.splitlines()
        for i, line in enumerate(lines):
            m = MD_HEADING.match(line)
            if not m:
                continue
            level = len(m.group("hash"))
            head_txt = m.group("text").strip()
            # Already has an id like '{#...}'?
            if re.search(r"\{\s*#[-a-z0-9]+?\s*\}\s*$", head_txt):
                if level <= max_depth:
                    id_match = re.search(
                        r"\{\s*#([-a-z0-9]+)\s*\}\s*$", head_txt
                    )
                    hid = (
                        id_match.group(1)
                        if id_match
                        else slugify_heading(
                            re.sub(r"\s*\{#.*\}\s*$", "", head_txt)
                        )
                    )
                    text_only = re.sub(
                        r"\s*\{#.*\}\s*$", "", head_txt
                    )
                    toc.append(
                        {"level": level, "text": text_only, "id": hid}
                    )
                continue
            hid = unique_id(slugify_heading(head_txt))
            lines[i] = f"{'#' * level} {head_txt} {{#{hid}}}"
            if level <= max_depth:
                toc.append({"level": level, "text": head_txt, "id": hid})
        return "\n".join(lines)

    text = _inject_atx_ids(text)

    if md_text.endswith("\n") and not text.endswith("\n"):
        text += "\n"
    return text, toc


def extract_markdown_attachments(
    cell, out_assets_dir
) -> Tuple[str, list[str]]:
    text = cell.get("source", "")
    atts = cell.get("attachments") or {}
    used = []

    def _repl(m):
        name = m.group("name")
        blob = atts.get(name)
        if not blob:
            return m.group(0)
        mime, b64 = next(iter(blob.items()))
        ext = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/svg+xml": ".svg",
            "image/webp": ".webp",
        }.get(mime, ".bin")
        data = base64.b64decode(b64)
        h = hashlib.sha256(data).hexdigest()[:8]
        fname = f"att-{slugify(name)}.{h}{ext}"
        ensure_dir(out_assets_dir)
        (out_assets_dir / fname).write_bytes(data)
        used.append(fname)
        return f"{ASSET_DIR_NAME}/{fname}"

    text = ATTACHMENT_URL.sub(_repl, text)
    return text, used
