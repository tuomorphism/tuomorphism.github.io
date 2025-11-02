import { getPermalink } from './utils/permalinks';

export const headerData = {
  links: [
    {
      text: 'About',
      href: getPermalink('/')
    },
    {
      text: 'Projects',
      href: getPermalink('/projects')
    },
    {
      text: 'Blog',
      href: getPermalink('/blog')
    },
  ],
  socials: [
    {
      ariaLabel: 'GitHub',
      icon: 'tabler:brand-github',
      href: 'https://github.com/tuomorphism',
    },
    {
      ariaLabel: 'LinkedIn',
      icon: 'tabler:brand-linkedin',
      href: 'https://www.linkedin.com/in/tuomorphism/',
    },
    {
      ariaLabel: 'Kaggle',
      icon: 'tabler:brand-kickstarter',
      href: 'https://www.kaggle.com/urjalacoder',
    },
  ],
};

export const footerData = {
  socialLinks: [
    { ariaLabel: 'Github', icon: 'tabler:brand-github', href: 'https://github.com/onwidget/astrowind' },
  ],
};
