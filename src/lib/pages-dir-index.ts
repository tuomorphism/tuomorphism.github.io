import { copyFile, readdir, stat } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { AstroIntegration } from 'astro';

/**
 * With `build.format: 'file'`, `/blog` is emitted as `blog.html` next to a `blog/` directory
 * of posts. Copy it to `blog/index.html` too, so the page resolves whether GitHub Pages
 * serves `blog.html` or redirects to the directory.
 */
export function pagesDirIndex(): AstroIntegration {
  return {
    name: 'pages-dir-index',
    hooks: {
      'astro:build:done': async ({ dir }) => {
        const walk = async (path: string): Promise<void> => {
          for (const entry of await readdir(path, { withFileTypes: true })) {
            if (!entry.isDirectory()) continue;
            const sub = join(path, entry.name);
            const page = `${sub}.html`;
            if (await stat(page).catch(() => null)) await copyFile(page, join(sub, 'index.html'));
            await walk(sub);
          }
        };
        await walk(fileURLToPath(dir));
      },
    },
  };
}
