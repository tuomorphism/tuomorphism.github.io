import path from 'path';
import { fileURLToPath } from 'url';

import { defineConfig } from 'astro/config';

import sitemap from '@astrojs/sitemap';
import tailwind from '@astrojs/tailwind';
import mdx from '@astrojs/mdx';
import partytown from '@astrojs/partytown';
import icon from 'astro-icon';
import compress from 'astro-compress';

import astrowind from './vendor/integration';

import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkHeadingId from 'remark-heading-id';

import rehypeRaw from 'rehype-raw';
import rehypeSlug from 'rehype-slug';
import rehypeKatex from 'rehype-katex';
import rehypeAutolinkHeadings from 'rehype-autolink-headings';

import {
  readingTimeRemarkPlugin,
  responsiveTablesRehypePlugin,
  lazyImagesRehypePlugin,
} from './src/utils/frontmatter';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const hasExternalScripts = false;
const whenExternalScripts = (items = []) =>
  hasExternalScripts ? (Array.isArray(items) ? items.map((i) => i()) : [items()]) : [];

export default defineConfig({
  output: 'static',

  integrations: [
    tailwind({ applyBaseStyles: false }),
    sitemap(),
    mdx(),
    icon({
      include: {
        tabler: ['*'],
        'flat-color-icons': [
          'template',
          'gallery',
          'approval',
          'document',
          'advertising',
          'currency-exchange',
          'voice-presentation',
          'business-contact',
          'database',
        ],
      },
    }),
    ...whenExternalScripts(() =>
      partytown({ config: { forward: ['dataLayer.push'] } })
    ),
    compress({
      CSS: true,
      HTML: { 'html-minifier-terser': { removeAttributeQuotes: false } },
      Image: false,
      JavaScript: true,
      SVG: false,
      Logger: 1,
    }),
    astrowind({ config: './src/config.yaml' }),
  ],

  image: { domains: ['cdn.pixabay.com'] },

  markdown: {
    remarkPlugins: [
      readingTimeRemarkPlugin,
      remarkGfm,
      remarkMath,
      remarkHeadingId, // supports # Title {#id}
    ],
    rehypePlugins: [
      // Parse raw HTML into the tree before other rehype plugins run
      [rehypeRaw, { passThrough: ['mdxJsxTextElement', 'mdxJsxFlowElement'] }],

      // Your custom/utility plugins (list separately, not nested in one array)
      responsiveTablesRehypePlugin,
      lazyImagesRehypePlugin,

      // Math rendering
      rehypeKatex,

      // Slug + autolink (rehypeSlug first)
      rehypeSlug,
      [rehypeAutolinkHeadings, { behavior: 'wrap' }],
    ],
  },

  vite: {
    resolve: {
      alias: { '~': path.resolve(__dirname, './src') },
    },
  },
});
