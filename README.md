# tuomorphism.github.io

Personal portfolio and blog, built with [Astro](https://astro.build) and Tailwind CSS.
Projects and blog posts are pulled from my other repositories at build time.

## Layout

```
content/
  sources.yml          repos to pull projects/posts from (+ per-post overrides)
  projects/*.yaml      hand-written projects (theses, Kaggle notebooks, ...)
  posts/**/*.md        hand-written posts
exporter/              Python: source repos -> generated/ + public/media/
generated/             exporter output (gitignored)
src/
  content.config.ts    the data model (projects, posts)
  lib/content.ts       queries: ordering, series, prev/next
  site.ts              site name, navigation, socials
  pages/  components/  layouts/  styles/
```

## Data model

- **Project**: `title`, `summary`, `date`, `featured`, `cover` (image/video URL), `links` (`label`, `url`, `kind` inferred from the URL), `tags`.
- **Post**: `title`, `description`, `publishDate`, `updatedDate`, `draft`, `project` (reference), `order`, `tags`, `source`.

Hand-written and generated entries share one schema. A post's `project` + `order` define a series: the project card lists
its posts, and posts link prev/next within the series. Post URLs are `/blog/<id>`; generated posts get
`<project>/<post>`.

## Design

All tokens live in `src/styles/global.css` (`@theme`) and are used through Tailwind utilities:

- **Colour**: neutrals `ink` > `text` > `muted` > `subtle`, plus `line`, `page`, `surface`, `sunken`; one accent
  (`accent`, `accent-strong`, `accent-soft`).
- **Type**: Inter for interface and headings, Source Serif 4 for reading text, JetBrains Mono for code.
- **Primitives**: `<Button>` (primary / secondary / ghost, `sm` / `md`, optional icon-only) for every button-like link;
  `.link` for inline text links; `.card` (+ `.card-interactive`) for every boxed surface; `.eyebrow` for small labels;
  `.container-page` for page width.
- **Posts**: `.post-body` styles rendered Markdown in one ~80-character column (text, code and figures alike); code
  cells and their notebook outputs are joined together, and over-wide equations are scaled down to fit. The table of
  contents is a sticky sidebar on wide screens, collapsible above the post otherwise.

## Source repos

A source repo can contain:

- `project.yml`: `title`, `description`, `tier` (1 = featured), `date`, `links`, optional `image` (cover path)
- `assets/hero.{mp4,webm,gif,png,jpg,webp}`: cover
- `blog/**/*.ipynb` / `blog/**/*.md`: posts, ordered by path. The first `# Heading` becomes the title, the first
  paragraph the description. Notebook metadata or Markdown frontmatter can set `title`, `description`, `publishDate`,
  `draft`, `tags`.
- Notebook cell tags: `remove-cell` / `remove-input` / `remove-output` leave things out; `hide-input` folds the code
  behind a "Show code" row and `show-input` keeps it open. Untagged code cells fold automatically when longer than 15
  lines or when they only do imports/setup, so posts read as prose with the code a click away.

Add the repo to `content/sources.yml` to include it.

## Development

```sh
npm install
pip install -r exporter/requirements.txt
npm run content     # clone/update source repos into .cache/ and export (optionally: -- --only <id>)
npm run dev
```

## Deployment

`.github/workflows/deploy.yaml` runs the exporter and builds the site on every push to `main`, nightly, on manual
dispatch, and on a `content-updated` repository dispatch (which source repos can send to publish right away).
Private source repos need the `EXTERNAL_GIT_PAT` secret.
