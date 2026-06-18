import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    // The JS suite loads Static/index.html in jsdom itself, so it runs in a
    // plain node environment. Keep collection scoped to the frontend tests.
    include: ['tests/frontend/**/*.test.js'],
    environment: 'node',
  },
});
