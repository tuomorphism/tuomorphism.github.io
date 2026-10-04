"""Builds site content from the source repositories listed in content/sources.yml.

Run from the repo root:  python3 -m exporter [--only ID ...]

Output (gitignored, rebuilt from scratch on each run):
  generated/projects/<id>.yaml
  generated/posts/<id>/<post>/index.md
  public/media/{projects,posts}/...      images, videos and notebook outputs
"""
