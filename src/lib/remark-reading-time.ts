import getReadingTime from 'reading-time';
import { toString } from 'mdast-util-to-string';
import type { Root } from 'mdast';

/** Exposes `minutesRead` via `render(entry).remarkPluginFrontmatter`. */
export function remarkReadingTime() {
  return (tree: Root, file: { data: { astro?: { frontmatter?: Record<string, unknown> } } }) => {
    const minutes = Math.max(1, Math.round(getReadingTime(toString(tree)).minutes));
    file.data.astro ??= {};
    file.data.astro.frontmatter ??= {};
    file.data.astro.frontmatter.minutesRead = minutes;
  };
}
