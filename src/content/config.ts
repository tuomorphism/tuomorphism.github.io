import { defineCollection, z } from 'astro:content';

const post = defineCollection({
  schema: z.object({
    title: z.string(),
    excerpt: z.string().optional(),    
    publishDate: z.coerce.date(),       // string dates -> Date
    updateDate: z.coerce.date().optional(),
    draft: z.boolean().default(false),
    tags: z.array(z.string()).default([]),
  }),
});

const projects = defineCollection({
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
