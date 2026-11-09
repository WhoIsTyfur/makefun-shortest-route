// Local scorer: node tools/score.mjs <input dir> <output dir>
// Validates each tour and prints its ratio to the nearest-neighbour tour, in millionths.
import { readdirSync, readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';

const [inDir, outDir] = process.argv.slice(2);
const d = (X, Y, a, b) => Math.round(Math.hypot(X[a] - X[b], Y[a] - Y[b]));

let sum = 0;
let valid = 0;
const files = readdirSync(inDir).filter((f) => f.endsWith('.txt')).sort();
for (const f of files) {
  const id = f.slice(0, -4);
  const lines = readFileSync(join(inDir, f), 'utf8').trim().split('\n');
  const n = parseInt(lines[0], 10);
  const X = new Float64Array(n);
  const Y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const [x, y] = lines[i + 1].trim().split(' ').map(Number);
    X[i] = x; Y[i] = y;
  }
  // reference: nearest neighbour from city 0, squared distance, lower index on tie
  const seen = new Uint8Array(n);
  let cur = 0;
  seen[0] = 1;
  let ref = 0;
  for (let s = 1; s < n; s++) {
    let best = -1;
    let bd = Infinity;
    for (let j = 0; j < n; j++) {
      if (seen[j]) continue;
      const dx = X[cur] - X[j];
      const dy = Y[cur] - Y[j];
      const d2 = dx * dx + dy * dy;
      if (d2 < bd) { bd = d2; best = j; }
    }
    ref += d(X, Y, cur, best);
    seen[best] = 1;
    cur = best;
  }
  ref += d(X, Y, cur, 0);

  const tp = join(outDir, id + '.tour');
  if (!existsSync(tp)) { console.log(id, 'MISSING'); continue; }
  const toks = readFileSync(tp, 'utf8').split(/\s+/).filter((s) => s.length > 0);
  const used = new Uint8Array(n);
  let ok = toks.length === n;
  const t = new Int32Array(n);
  for (let i = 0; ok && i < n; i++) {
    if (!/^(0|[1-9][0-9]*)$/.test(toks[i])) { ok = false; break; }
    const v = Number(toks[i]);
    if (v >= n || used[v]) { ok = false; break; }
    used[v] = 1;
    t[i] = v;
  }
  if (!ok) { console.log(id, 'INVALID'); continue; }
  let len = 0;
  for (let i = 0; i < n; i++) len += d(X, Y, t[i], t[(i + 1) % n]);
  const ratio = len / ref;
  sum += ratio;
  valid++;
  console.log(id, 'n=' + n, 'len=' + len, 'nn=' + ref, 'ratio=' + Math.round(ratio * 1e6));
}
console.log('valid', valid + '/' + files.length, 'score', Math.round((sum / files.length) * 1e6));
