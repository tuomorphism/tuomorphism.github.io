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
import { pagesDirIndex } from './src/lib/pages-dir-index';

export default defineConfig({
  site: 'https://tuomorphism.github.io',
  trailingSlash: 'never',
  // Emit /projects.html rather than /projects/index.html: GitHub Pages would otherwise
  // 301-redirect every link to its trailing-slash form.
  build: { format: 'file' },

  integrations: [sitemap(), icon(), pagesDirIndex()],

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

  // Old URLs: the former blog index and post URLs from the previous exporter.
  redirects: {
    // The post list is the home page.
    '/blog': '/',
    '/blog/diffusion-on-the-edge-01-introduction-01-introduction': '/blog/diffusion-on-the-edge/01-introduction',
    '/blog/diffusion-on-the-edge-02-maximal-entropy-02-maximal-learning':
      '/blog/diffusion-on-the-edge/02-maximal-learning',
    '/blog/drone-sim-nav-drone-simulation': '/blog/drone-navigation-sim/drone-simulation',
  },

  vite: {
    plugins: [tailwindcss()],
  },
});
