/**
 * Data model.
 *
 * Both collections merge hand-written entries (content/, committed) with entries
 * produced by the exporter (generated/, rebuilt on every deploy from the repos
 * listed in content/sources.yml). Both sources share the same schema.
 *
 *   projects  content/projects/<id>.yaml | generated/projects/<id>.yaml
 *   posts     content/posts/<id>.md      | generated/posts/<project>/<post>/index.md
 *
 * Posts link to their project with `project`. Posts of the same project form a
 * series ordered by `order` (then publish date); prev/next and the per-project
 * post lists are derived from that at build time (src/lib/content.ts).
 */
import { defineCollection, reference } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const linkKinds = ['repo', 'demo', 'pdf', 'notebook', 'other'] as const;

function inferLinkKind(label: string, url: string): (typeof linkKinds)[number] {
  const l = label.toLowerCase();
  // The URL is more reliable than the label ("Code" may point at Kaggle), so check it first.
  if (/github\.com|gitlab\.com/.test(url)) return 'repo';
  if (/kaggle\.com|colab\.research/.test(url)) return 'notebook';
  if (url.endsWith('.pdf')) return 'pdf';
  if (['repo', 'code', 'source'].includes(l)) return 'repo';
  if (['notebook'].includes(l)) return 'notebook';
  if (['pdf', 'paper', 'thesis'].includes(l)) return 'pdf';
  if (['demo', 'live', 'site'].includes(l)) return 'demo';
  return 'other';
}

const link = z
  .object({
    label: z.string(),
    url: z.string(),
    kind: z.enum(linkKinds).optional(),
  })
  .transform((l) => ({ ...l, kind: l.kind ?? inferLinkKind(l.label, l.url) }));

/** `content/posts/a/b.md` and `generated/posts/a/b/index.md` both become `a/b`. */
const postId = ({ entry }: { entry: string }) =>
  entry
    .replace(/^(content|generated)\/posts\//, '')
    .replace(/\/index\.md$/, '')
    .replace(/\.md$/, '');

const projectId = ({ entry }: { entry: string }) => entry.replace(/^.*\//, '').replace(/\.ya?ml$/, '');

const projects = defineCollection({
  loader: glob({
    base: '.',
    pattern: ['content/projects/*.{yaml,yml}', 'generated/projects/*.yaml'],
    generateId: projectId,
  }),
  schema: z.object({
    title: z.string(),
    summary: z.string(),
    /** Featured projects are shown first, with their cover media. */
    featured: z.boolean().default(false),
    /** Date the project was last meaningfully worked on; used for ordering. */
    date: z.coerce.date(),
    /** Image or video URL (videos are detected by extension). */
    cover: z.string().optional(),
    links: z.array(link).default([]),
    tags: z.array(z.string()).default([]),
  }),
});

const posts = defineCollection({
  loader: glob({
    base: '.',
    pattern: ['content/posts/**/*.md', 'generated/posts/**/index.md'],
    generateId: postId,
  }),
  schema: z.object({
    title: z.string(),
    description: z.string().optional(),
    publishDate: z.coerce.date(),
    updatedDate: z.coerce.date().optional(),
    draft: z.boolean().default(false),
    project: reference('projects').optional(),
    /** Position within the project's series of posts. */
    order: z.number().optional(),
    tags: z.array(z.string()).default([]),
    /** Link to the original notebook / markdown file. */
    source: z.url().optional(),
  }),
});

export const collections = { projects, posts };
