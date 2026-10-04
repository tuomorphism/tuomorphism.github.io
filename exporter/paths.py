from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES_FILE = ROOT / "content" / "sources.yml"
CACHE_DIR = ROOT / ".cache" / "sources"
GENERATED_DIR = ROOT / "generated"
MEDIA_DIR = ROOT / "public" / "media"
MEDIA_URL = "/media"
