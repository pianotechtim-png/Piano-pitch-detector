import { JSDOM } from 'jsdom';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const htmlPath = resolve(here, '../../Static/index.html');

// Load the *real* served page and run its inline scripts, so the tests call the
// exact functions that ship -- no copied logic that could silently drift.
// The page's top-level functions (centsColor, computeDrift, ...) become
// properties on `window`, which is what we return.
export function loadApp() {
  const html = readFileSync(htmlPath, 'utf8');
  const dom = new JSDOM(html, { runScripts: 'dangerously', url: 'http://localhost/' });
  return dom.window;
}

// Parse the hue out of an "hsl(H,75%,55%)" string for assertions.
export function hueOf(hsl) {
  const m = /^hsl\((\d+(?:\.\d+)?),/.exec(hsl);
  return m ? parseFloat(m[1]) : null;
}
