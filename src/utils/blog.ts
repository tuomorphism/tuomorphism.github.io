import type { CollectionEntry } from 'astro:content';
import { getCollection, render } from 'astro:content';
import type { Post } from '~/types';
import { APP_BLOG } from 'astrowind:config';
import { cleanSlug, trimSlash, POST_PERMALINK_PATTERN } from './permalinks';

const normalizeSlug = (s: string) => trimSlash(s).replace(/\/index$/i, '');

const generatePermalink = async ({
  id,
  slug,
  publishDate,
  category,
}: {
  id: string;
  slug: string;
  publishDate: Date;
  category: string | undefined;
}) => {
  const year = String(publishDate.getFullYear()).padStart(4, '0');
  const month = String(publishDate.getMonth() + 1).padStart(2, '0');
  const day = String(publishDate.getDate()).padStart(2, '0');
  const hour = String(publishDate.getHours()).padStart(2, '0');
  const minute = String(publishDate.getMinutes()).padStart(2, '0');
  const second = String(publishDate.getSeconds()).padStart(2, '0');

  const permalink = POST_PERMALINK_PATTERN.replace('%slug%', slug)
    .replace('%id%', id)
    .replace('%category%', category || '')
    .replace('%year%', year)
    .replace('%month%', month)
    .replace('%day%', day)
    .replace('%hour%', hour)
    .replace('%minute%', minute)
    .replace('%second%', second);

  return permalink
    .split('/')
    .map((el) => trimSlash(el))
    .filter(Boolean)
    .join('/');
};

const getNormalizedPost = async (post: CollectionEntry<'post'>): Promise<Post> => {
  const { id, data } = post;
  const { Content, remarkPluginFrontmatter } = await render(post);

  const {
    frontSlug,
    publishDate: rawPublishDate = new Date(),
    updateDate: rawUpdateDate,
    title,
    excerpt,
    image,
    category: rawCategory,
    author,
    draft = false,
    metadata = {},
  } = data as any;

  if (!frontSlug || typeof frontSlug !== 'string') {
    throw new Error(`Frontmatter "slug" is required for post ${id}`);
  }

  const slug = normalizeSlug(frontSlug);
  const publishDate = new Date(rawPublishDate);
  const updateDate = rawUpdateDate ? new Date(rawUpdateDate) : undefined;

  const category = rawCategory
    ? { slug: cleanSlug(rawCategory), title: rawCategory }
    : undefined;

  return {
    id,
    slug,
    permalink: await generatePermalink({ id, slug, publishDate, category: category?.slug }),
    publishDate,
    updateDate,
    title,
    excerpt,
    image,
    category,
    author,
    draft,
    metadata,
    Content,
    readingTime: remarkPluginFrontmatter?.readingTime,
  };
};

const load = async (): Promise<Array<Post>> => {
  const posts = await getCollection('post');
  const normalized = await Promise.all(posts.map(getNormalizedPost));

  // Optional: guard against duplicate slugs
  const seen = new Set<string>();
  for (const p of normalized) {
    if (seen.has(p.slug)) throw new Error(`Duplicate slug "${p.slug}"`);
    seen.add(p.slug);
  }

  return normalized
    .sort((a, b) => b.publishDate.valueOf() - a.publishDate.valueOf())
    .filter((p) => !p.draft);
};

let _posts: Array<Post>;

export const isBlogEnabled = APP_BLOG.isEnabled;
export const isRelatedPostsEnabled = APP_BLOG.isRelatedPostsEnabled;
export const isBlogListRouteEnabled = APP_BLOG.list.isEnabled;
export const isBlogPostRouteEnabled = APP_BLOG.post.isEnabled;
export const isBlogCategoryRouteEnabled = APP_BLOG.category.isEnabled;
export const isBlogTagRouteEnabled = APP_BLOG.tag.isEnabled;

export const blogListRobots = APP_BLOG.list.robots;
export const blogPostRobots = APP_BLOG.post.robots;
export const blogCategoryRobots = APP_BLOG.category.robots;
export const blogTagRobots = APP_BLOG.tag.robots;

export const fetchPosts = async (): Promise<Array<Post>> => {
  if (!_posts) _posts = await load();
  return _posts;
};
