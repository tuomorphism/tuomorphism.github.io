import { defineCollection, z } from 'astro:content';

const linkObject = z.object({
  title: z.string(),
  url: z.string()
});

const post = defineCollection({
  type: 'content',
  schema: z.object({
    // Data for displaying the post
    title: z.string(),
    excerpt: z.string().optional(),
    publishDate: z.coerce.date(),
    updateDate: z.coerce.date().optional(),
    draft: z.boolean().default(false),
    frontSlug: z.string(),

    // linking between posts
    next: linkObject.optional(),   // slug of the “next” post
    prev: linkObject.optional(),   // slug of the “previous” post
  }),
});

const projects = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    description: z.string(),
    tier: z.number().default(2),
    image: z.string().optional(),
    date: z.date().optional(),
    links: z.array(linkObject).optional(),
    blog_posts: z.array(linkObject).optional()
  })
});

export const collections = { post, projects };
