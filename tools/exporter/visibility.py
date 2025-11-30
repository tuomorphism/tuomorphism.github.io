from __future__ import annotations

import copy

from nbformat import NotebookNode

from .utils import _norm_text

_HIDDEN_INPUT_TAGS = {"hide-input", "remove-input", "hide_input", "remove_input"}
_HIDDEN_OUTPUT_TAGS = {"hide-output", "remove-output", "hide_output", "remove_output"}
_REMOVE_CELL_TAGS = {"remove-cell", "hide-cell", "remove_cell", "hide_cell"}


def _tags(cell: NotebookNode) -> set:
    md = getattr(cell, "metadata", {}) or {}
    return set((md.get("tags") or []))


def _is_markdown_effectively_empty(cell: NotebookNode) -> bool:
    src = _norm_text(cell.get("source", "")).strip()
    attachments = (cell.get("attachments") or {})
    return (src == "") and (not attachments)


def _is_code_effectively_empty(cell: NotebookNode) -> bool:
    src = _norm_text(cell.get("source", "")).strip()
    outputs = cell.get("outputs") or []
    return (src == "") and (len(outputs) == 0)


def _apply_hidden_flags(cell: NotebookNode):
    c = copy.deepcopy(cell)
    md = getattr(c, "metadata", None)
    if md is None:
        from nbformat import NotebookNode as NBNode

        c.metadata = md = NBNode()

    jup = md.get("jupyter") if isinstance(md.get("jupyter"), dict) else {}
    tags = _tags(c)

    if tags & _REMOVE_CELL_TAGS:
        return None

    source_hidden = (
        bool(jup.get("source_hidden"))
        or bool(tags & _HIDDEN_INPUT_TAGS)
        or bool(md.get("source_hidden"))
    )
    if source_hidden and c.get("cell_type") in {"code", "markdown"}:
        if c.get("cell_type") == "markdown":
            return None
        if c.get("cell_type") == "code":
            c["source"] = ""

    outputs_hidden = (
        bool(jup.get("outputs_hidden"))
        or bool(tags & _HIDDEN_OUTPUT_TAGS)
        or bool(md.get("outputs_hidden"))
    )
    if outputs_hidden and c.get("cell_type") == "code":
        c["outputs"] = []
        c["execution_count"] = None

    return c


def filter_and_apply_visibility(nbnode) -> None:
    new_cells = []
    for cell in nbnode.cells:
        cell2 = _apply_hidden_flags(cell)
        if cell2 is None:
            continue

        if cell2.get("cell_type") == "markdown":
            if _is_markdown_effectively_empty(cell2):
                continue
        elif cell2.get("cell_type") == "code":
            if _is_code_effectively_empty(cell2):
                continue

        new_cells.append(cell2)

    nbnode.cells = new_cells
