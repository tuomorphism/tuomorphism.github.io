import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import sitemap from '@astrojs/sitemap';
import tailwindcss from '@tailwindcss/vite';
import icon from 'astro-icon';

import remarkMath from 'remark-math';
import rehypeRaw from 'rehype-raw';
import rehypeKatex from 'rehype-katex';
import rehypeAutolinkHeadings from 'rehype-autolink-headings';

import { remarkReadingTime } from './src/lib/remark-reading-time';

export default defineConfig({
  site: 'https://tuomorphism.github.io',
  trailingSlash: 'never',

  integrations: [sitemap(), icon()],

  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath, remarkReadingTime],
      rehypePlugins: [
        // Notebook exports contain raw HTML (outputs, videos); parse it into the tree.
        rehypeRaw,
        rehypeKatex,
        [rehypeAutolinkHeadings, { behavior: 'wrap' }],
      ],
    }),
    shikiConfig: { theme: 'github-light' },
  },

  // Post URLs used by the previous exporter.
  redirects: {
    '/blog/diffusion-on-the-edge-01-introduction-01-introduction': '/blog/diffusion-on-the-edge/01-introduction',
    '/blog/diffusion-on-the-edge-02-maximal-entropy-02-maximal-learning':
      '/blog/diffusion-on-the-edge/02-maximal-learning',
    '/blog/drone-sim-nav-drone-simulation': '/blog/drone-navigation-sim/drone-simulation',
  },

  vite: {
    plugins: [tailwindcss()],
  },
});
