/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{vue,js,ts}'],
  theme: {
    extend: {
      colors: {
        // Claude 风格配色
        'claude-bg': '#FAF9F7',
        'claude-surface': '#FFFFFF',
        'claude-border': '#E8E5E0',
        'claude-text': '#1A1A1A',
        'claude-text-secondary': '#6B6B6B',
        'claude-accent': '#D97706',
        'claude-hover': '#F5F3EF',
      },
      borderRadius: {
        'xl': '1rem',
        '2xl': '1.5rem',
      },
    },
  },
  plugins: [],
}
