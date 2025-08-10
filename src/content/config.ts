import { defineCollection, z } from 'astro:content';

const blog = defineCollection({
  schema: z.object({
    title: z.string(),
    description: z.string(),
    date: z.date(),
    draft: z.boolean().optional(),
    tags: z.array(z.string()).optional()
  })
});

const projects = defineCollection({
  schema: z.object({
    title: z.string(),
    description: z.string(),
    tier: z.number().default(2),
    image: z.string().optional(),
    date: z.date().optional(),
    links: z.array(z.object({ label: z.string(), url: z.string() })).optional()
  })
});

export const collections = { blog, projects };
