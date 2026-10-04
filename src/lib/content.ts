import { getCollection, type CollectionEntry } from 'astro:content';

export type Project = CollectionEntry<'projects'>;
export type Post = CollectionEntry<'posts'>;

export const postUrl = (post: Post) => `/blog/${post.id}`;

/** Featured first, then most recent, then by title. */
export async function getProjects(): Promise<Project[]> {
  const projects = await getCollection('projects');
  return projects.sort(
    (a, b) =>
      Number(b.data.featured) - Number(a.data.featured) ||
      b.data.date.valueOf() - a.data.date.valueOf() ||
      a.data.title.localeCompare(b.data.title)
  );
}

/** Published posts (drafts only in dev), newest first. */
export async function getPosts(): Promise<Post[]> {
  const posts = await getCollection('posts', ({ data }) => import.meta.env.DEV || !data.draft);
  return posts.sort(
    (a, b) => b.data.publishDate.valueOf() - a.data.publishDate.valueOf() || a.data.title.localeCompare(b.data.title)
  );
}

/** A project's posts in reading order. */
export function seriesOf(projectId: string, posts: Post[]): Post[] {
  return posts
    .filter((p) => p.data.project?.id === projectId)
    .sort(
      (a, b) =>
        (a.data.order ?? Infinity) - (b.data.order ?? Infinity) ||
        a.data.publishDate.valueOf() - b.data.publishDate.valueOf()
    );
}

export function neighbours(post: Post, posts: Post[]): { prev?: Post; next?: Post } {
  if (!post.data.project) return {};
  const series = seriesOf(post.data.project.id, posts);
  const i = series.findIndex((p) => p.id === post.id);
  return { prev: series[i - 1], next: series[i + 1] };
}

export const isVideo = (url: string) => /\.(mp4|webm)$/i.test(url);

export const formatDate = (d: Date) =>
  d.toLocaleDateString('en-GB', { year: 'numeric', month: 'short', day: 'numeric', timeZone: 'UTC' });

/** The page's public path: with `build.format: 'file'` Astro reports `/about.html` and `/index.html` during builds. */
export const pagePath = (url: URL) => url.pathname.replace(/(\/index)?\.html$/, '').replace(/\/$/, '') || '/';
