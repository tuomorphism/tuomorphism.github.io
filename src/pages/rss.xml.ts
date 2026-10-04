import rss from '@astrojs/rss';
import type { APIContext } from 'astro';
import { getPosts, postUrl } from '~/lib/content';
import { site } from '~/site';

export async function GET(context: APIContext) {
  const posts = await getPosts();
  return rss({
    title: site.name,
    description: site.description,
    site: context.site!,
    items: posts.map((post) => ({
      link: postUrl(post),
      title: post.data.title,
      description: post.data.description,
      pubDate: post.data.publishDate,
    })),
  });
}
