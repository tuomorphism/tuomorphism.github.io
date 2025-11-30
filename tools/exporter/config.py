#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import re

# ---------- Paths

# This assumes export_config.py sits in tools/ at the repo root.
ROOT = pathlib.Path(__file__).resolve().parents[2]
EXTERNAL = ROOT / "external"
TEMPLATE_DIR = ROOT / "tools" / "templates"
POST_OUT = ROOT / "src" / "content" / "post"
PROJ_OUT = ROOT / "src" / "content" / "projects"
PUBLIC_BLOG = ROOT / "public" / "blog"
PUBLIC_PROJECTS = ROOT / "public" / "projects"

# ---------- Config

ASSET_DIR_NAME = "assets"
ASSET_SOURCE_DIR_CANDIDATES = ("assets", "_assets")
MAX_TOC_DEPTH = 3

# Some shared regexes

MD_LINK_IMG = re.compile(
    r'(!?)\[(?P<alt>[^\]]*)\]\((?P<url>[^)\s]+)(?:\s+"[^"]*")?\)'
)
HTML_SRC_OR_HREF = re.compile(
    r'(?P<attr>\bsrc\b|\bhref\b)\s*=\s*([\'"])(?P<url>[^\'"]+)\2'
)
MD_HEADING = re.compile(r'^(?P<hash>#{1,6})\s+(?P<text>.+?)\s*$',
                        re.MULTILINE)
SETEXT_RE = re.compile(
    r'^(?P<text>.+?)\n(?P<underline>=+|-+)\s*$', re.MULTILINE
)
ATTACHMENT_URL = re.compile(r'\battachment:(?P<name>[^)\s]+)')
BLOCK_HTML = re.compile(
    r'^(<(?P<tag>(div|table|figure|video|iframe|details|summary|blockquote)\b)'
    r'[\s\S]*?>[\s\S]*?</(?P=tag)>)$',
    re.MULTILINE,
)
FENCE = re.compile(r"(^```.*?$)(.*?)(^```$)",
                   re.MULTILINE | re.DOTALL)
INLINE_MATH = re.compile(r'(?<!\\)\$(.+?)(?<!\\)\$')
BLOCK_MATH = re.compile(
    r'(^\$\$.*?^\$\$)', re.MULTILINE | re.DOTALL
)
SPACES_EOL = re.compile(r'[ \t]+$', re.MULTILINE)
SLUG_RE = re.compile(r"[^a-z0-9-]+")
