import { defineCollection, z } from 'astro:content';
const post = defineCollection({
  type: 'content',
  schema: z.object({
    // Data for displaying the post
    title: z.string(),
    excerpt: z.string().optional(),
    publishDate: z.coerce.date(),
    updateDate: z.coerce.date().optional(),
    draft: z.boolean().default(false),
    tags: z.array(z.string()).default([]),

    // linking between posts
    next: z.string().optional(),   // slug of the “next” post
    prev: z.string().optional(),   // slug of the “previous” post
    slug: z.string().optional(),
  })
});

const projects = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    description: z.string(),
    tier: z.number().default(2),
    image: z.string().optional(),
    date: z.date().optional(),
    links: z.array(z.object({ label: z.string(), url: z.string() })).optional(),
    blog_posts: z.array(z.object({
      title: z.string(),
      url: z.string()
    })).optional()
  })
});

export const collections = { post, projects };
