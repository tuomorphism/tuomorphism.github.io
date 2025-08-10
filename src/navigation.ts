import { getPermalink } from './utils/permalinks';

export const headerData = {
  links: [
    {
      text: 'Projects',
      href: getPermalink('/projects')
    },
    {
      text: 'Blog',
      href: getPermalink('/blog')
    },
    {
      text: 'CV',
      href: getPermalink('/cv')
    },
  ],
};

export const footerData = {
  socialLinks: [
    { ariaLabel: 'Github', icon: 'tabler:brand-github', href: 'https://github.com/onwidget/astrowind' },
  ],
};
