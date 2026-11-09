// shortest-route-01 entry: greedy-edge start tour, 2-opt + Or-opt local search with
// neighbour lists and don't-look bits, then iterated local search with local segment-swap
// kicks undone through a reversal journal. Work is bounded by fixed iteration counts and
// a constant-seed PRNG, so both runs write identical bytes.
import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const K = 10; // neighbour list size
// ILS kicks per city and whether to use the LK chain; argv overrides are for local tuning
// only (the grader passes no arguments).
const KICKS_PER_CITY = process.argv[2] ? Number(process.argv[2]) : 45; // ~89 s at 60 in the 1-CPU sandbox
const USE_LK = process.argv[3] ? process.argv[3] === '1' : false;
const KICK_LEN = process.argv[4] ? Number(process.argv[4]) : 50; // max kick segment length

let n = 0;
let X = new Float64Array(0);
let Y = new Float64Array(0);
let tour = new Int32Array(0);
let pos = new Int32Array(0);
let neigh = new Int32Array(0);
let queue = new Int32Array(0);
let inQ = new Uint8Array(0);
let qHead = 0;
let qTail = 0;
let qCount = 0;
let journal = [];
let journaling = false;
let rngState = 0;

function rand() {
  // xorshift32, deterministic
  let x = rngState;
  x ^= x << 13; x >>>= 0;
  x ^= x >>> 17;
  x ^= x << 5; x >>>= 0;
  rngState = x;
  return x;
}

function dist(a, b) {
  const dx = X[a] - X[b];
  const dy = Y[a] - Y[b];
  return Math.round(Math.sqrt(dx * dx + dy * dy));
}

function next(c) {
  const p = pos[c] + 1;
  return tour[p === n ? 0 : p];
}

function prev(c) {
  const p = pos[c] - 1;
  return tour[p < 0 ? n - 1 : p];
}

// Reverse the tour between positions i and j inclusive (forward, wrapping), choosing the
// shorter side of the cycle.
function reverse(i, j) {
  let len = j - i;
  if (len < 0) len += n;
  len += 1;
  if (len * 2 > n) {
    const ni = j + 1;
    const nj = i - 1;
    i = ni >= n ? 0 : ni;
    j = nj < 0 ? n - 1 : nj;
    len = n - len;
  }
  if (journaling) journal.push(i, j);
  for (let s = len >> 1; s > 0; s--) {
    const ci = tour[i];
    const cj = tour[j];
    tour[i] = cj; pos[cj] = i;
    tour[j] = ci; pos[ci] = j;
    i++; if (i === n) i = 0;
    j--; if (j < 0) j = n - 1;
  }
}

// Remove edges (a,b) and (c,d), add (a,c) and (b,d). b follows a in the same direction
// that d follows c.
function move2(a, b, c, d) {
  if (next(a) === b) reverse(pos[b], pos[c]);
  else reverse(pos[a], pos[d]);
}

function push(c) {
  if (inQ[c]) return;
  inQ[c] = 1;
  queue[qTail] = c;
  qTail++; if (qTail === n) qTail = 0;
  qCount++;
}

function pop() {
  const c = queue[qHead];
  qHead++; if (qHead === n) qHead = 0;
  qCount--;
  inQ[c] = 0;
  return c;
}

function try2opt(a) {
  for (let dir = 0; dir < 2; dir++) {
    const b = dir === 0 ? next(a) : prev(a);
    const dab = dist(a, b);
    const base = a * K;
    for (let k = 0; k < K; k++) {
      const c = neigh[base + k];
      const dac = dist(a, c);
      if (dac >= dab) break;
      const d = dir === 0 ? next(c) : prev(c);
      if (c === b || d === a) continue;
      const delta = dac + dist(b, d) - dab - dist(c, d);
      if (delta < 0) {
        move2(a, b, c, d);
        push(a); push(b); push(c); push(d);
        return delta;
      }
    }
  }
  return 0;
}

function inSeg(c, ps, len) {
  let o = pos[c] - ps;
  if (o < 0) o += n;
  return o < len;
}

function tryOrOpt(a) {
  for (let len = 1; len <= 3; len++) {
    for (let side = 0; side < 2; side++) {
      if (len === 1 && side === 1) continue;
      let ps = side === 0 ? pos[a] : pos[a] - (len - 1);
      if (ps < 0) ps += n;
      const s = tour[ps];
      let pe = ps + len - 1;
      if (pe >= n) pe -= n;
      const e = tour[pe];
      const p = prev(s);
      const nx = next(e);
      const g = dist(p, s) + dist(e, nx) - dist(p, nx);
      if (g <= 0) continue;
      for (let end = 0; end < 2; end++) {
        const x = end === 0 ? s : e;
        if (end === 1 && len === 1) break;
        const base = x * K;
        for (let k = 0; k < K; k++) {
          const c = neigh[base + k];
          const dxc = dist(x, c);
          if (dxc >= g) break;
          if (inSeg(c, ps, len)) continue;
          for (let w = 0; w < 2; w++) {
            const u = w === 0 ? c : prev(c);
            const v = w === 0 ? next(c) : c;
            if (inSeg(u, ps, len) || inSeg(v, ps, len)) continue;
            if (v === p || u === nx) continue;
            const duv = dist(u, v);
            const add1 = dist(u, s) + dist(e, v);
            const add2 = dist(u, e) + dist(s, v);
            const add = add1 < add2 ? add1 : add2;
            const delta = add - duv - g;
            if (delta < 0) {
              move2(p, s, u, v);
              move2(p, u, nx, e);
              if (add1 < add2 && len > 1) move2(u, e, s, v);
              push(p); push(nx); push(s); push(e); push(u); push(v);
              return delta;
            }
          }
        }
      }
    }
  }
  return 0;
}

function undoTo(mark) {
  const was = journaling;
  journaling = false;
  for (let q = journal.length - 2; q >= mark; q -= 2) reverse(journal[q], journal[q + 1]);
  journal.length = mark;
  journaling = was;
}

// Lin-Kernighan style chain of sequential 2-opt moves. t1 is fixed, (t1,t2) is the edge
// to remove, g the gain so far without the closing edge. Moves are applied tentatively and
// rolled back through the journal; returns the gain of the first improving closed tour.
const BREADTH = [6, 4, 2, 2, 1, 1, 1, 1];
const MAX_DEPTH = 7;
const LK_SPAN = process.argv[5] ? Number(process.argv[5]) : 1000;
function lkStep(t1, t2, g, level) {
  const fwd = next(t1) === t2;
  const base = t2 * K;
  const width = BREADTH[level];
  let tried = 0;
  for (let k = 0; k < K && tried < width; k++) {
    const t3 = neigh[base + k];
    const g1 = g - dist(t2, t3);
    if (g1 <= 0) break;
    if (t3 === t1) continue;
    const t4 = fwd ? prev(t3) : next(t3);
    if (t4 === t2 || t4 === t1) continue;
    tried++;
    const g2 = g1 + dist(t3, t4);
    const close = g2 - dist(t4, t1);
    if (close <= 0) {
      // only explore deeper through cheap (short) tentative reversals
      let span = fwd ? pos[t4] - pos[t2] : pos[t2] - pos[t4];
      if (span < 0) span += n;
      if (span > n - span) span = n - span;
      if (span > LK_SPAN || level + 1 >= MAX_DEPTH) continue;
    }
    const mark = journal.length;
    move2(t1, t2, t4, t3);
    if (close > 0) {
      push(t2); push(t3); push(t4);
      return close;
    }
    if (level + 1 < MAX_DEPTH) {
      const r = lkStep(t1, t4, g2, level + 1);
      if (r > 0) {
        push(t2); push(t3); push(t4);
        return r;
      }
    }
    undoTo(mark);
  }
  return 0;
}

function tryLK(t1) {
  for (let dir = 0; dir < 2; dir++) {
    const t2 = dir === 0 ? next(t1) : prev(t1);
    const r = lkStep(t1, t2, dist(t1, t2), 0);
    if (r > 0) {
      push(t1);
      return -r;
    }
  }
  return 0;
}

function localSearch() {
  let total = 0;
  while (qCount > 0) {
    const a = pop();
    let d = USE_LK ? tryLK(a) : try2opt(a);
    if (d === 0) d = tryOrOpt(a);
    total += d;
  }
  return total;
}

function buildNeighbours() {
  let maxC = 1;
  for (let i = 0; i < n; i++) {
    if (X[i] > maxC) maxC = X[i];
    if (Y[i] > maxC) maxC = Y[i];
  }
  const G = Math.max(1, Math.floor(Math.sqrt(n / 2)));
  const cell = (maxC + 1) / G;
  const cellOf = new Int32Array(n);
  const start = new Int32Array(G * G + 1);
  for (let i = 0; i < n; i++) {
    let gx = Math.floor(X[i] / cell); if (gx >= G) gx = G - 1;
    let gy = Math.floor(Y[i] / cell); if (gy >= G) gy = G - 1;
    cellOf[i] = gy * G + gx;
    start[cellOf[i] + 1]++;
  }
  for (let c = 0; c < G * G; c++) start[c + 1] += start[c];
  const fill = start.slice(0, G * G);
  const items = new Int32Array(n);
  for (let i = 0; i < n; i++) items[fill[cellOf[i]]++] = i;

  const kk = Math.min(K, n - 1);
  neigh = new Int32Array(n * K).fill(-1);
  const bd = new Float64Array(kk);
  const bi = new Int32Array(kk);
  for (let i = 0; i < n; i++) {
    let cnt = 0;
    const cx = cellOf[i] % G;
    const cy = (cellOf[i] - cx) / G;
    for (let r = 0; r <= G; r++) {
      for (let gy = cy - r; gy <= cy + r; gy++) {
        if (gy < 0 || gy >= G) continue;
        const ring = gy === cy - r || gy === cy + r;
        for (let gx = cx - r; gx <= cx + r; gx += ring ? 1 : 2 * r) {
          if (gx >= 0 && gx < G) {
            const cc = gy * G + gx;
            for (let t = start[cc]; t < start[cc + 1]; t++) {
              const j = items[t];
              if (j === i) continue;
              const dx = X[i] - X[j];
              const dy = Y[i] - Y[j];
              const d2 = dx * dx + dy * dy;
              if (cnt === kk && (d2 > bd[kk - 1] || (d2 === bd[kk - 1] && j > bi[kk - 1]))) continue;
              let q = cnt < kk ? cnt++ : kk - 1;
              while (q > 0 && (bd[q - 1] > d2 || (bd[q - 1] === d2 && bi[q - 1] > j))) {
                bd[q] = bd[q - 1]; bi[q] = bi[q - 1]; q--;
              }
              bd[q] = d2; bi[q] = j;
            }
          }
          if (r === 0) break;
        }
      }
      if (cnt === kk) {
        const reach = r * cell;
        if (bd[kk - 1] <= reach * reach) break;
      }
    }
    for (let k = 0; k < cnt; k++) neigh[i * K + k] = bi[k];
    // pad short lists by repeating the last neighbour (only when n - 1 < K)
    for (let k = cnt; k < K; k++) neigh[i * K + k] = bi[cnt - 1];
  }
}

function greedyTour() {
  // candidate edges from neighbour lists
  const m = n * K;
  const ea = new Int32Array(m);
  const eb = new Int32Array(m);
  const ed = new Float64Array(m);
  let ne = 0;
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < K; k++) {
      const j = neigh[i * K + k];
      if (j > i || (j < i && !hasNeighbour(j, i))) {
        ea[ne] = i; eb[ne] = j; ed[ne] = dist(i, j); ne++;
      }
    }
  }
  const order = new Int32Array(ne);
  for (let t = 0; t < ne; t++) order[t] = t;
  order.sort((p, q) => ed[p] - ed[q] || ea[p] - ea[q] || eb[p] - eb[q]);

  const parent = new Int32Array(n);
  for (let i = 0; i < n; i++) parent[i] = i;
  const find = (x) => {
    while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
  };
  const deg = new Uint8Array(n);
  const adj = new Int32Array(2 * n).fill(-1);
  let added = 0;
  for (let t = 0; t < ne && added < n - 1; t++) {
    const e = order[t];
    const a = ea[e];
    const b = eb[e];
    if (deg[a] >= 2 || deg[b] >= 2) continue;
    const ra = find(a);
    const rb = find(b);
    if (ra === rb) continue;
    parent[ra] = rb;
    adj[2 * a + deg[a]++] = b;
    adj[2 * b + deg[b]++] = a;
    added++;
  }

  // endpoints of fragments (a lone city appears once)
  const ends = [];
  const endIdx = new Int32Array(n).fill(-1);
  for (let i = 0; i < n; i++) {
    if (deg[i] < 2) { endIdx[i] = ends.length; ends.push(i); }
  }
  const removeEnd = (c) => {
    const k = endIdx[c];
    if (k < 0) return;
    const last = ends.pop();
    if (last !== c) { ends[k] = last; endIdx[last] = k; }
    endIdx[c] = -1;
  };

  tour = new Int32Array(n);
  let t = 0;
  let cur = ends.length > 0 ? ends[0] : 0;
  while (true) {
    removeEnd(cur);
    let prv = -1;
    let c = cur;
    while (true) {
      tour[t++] = c;
      const a0 = adj[2 * c];
      const a1 = adj[2 * c + 1];
      const nx = a0 !== prv && a0 >= 0 ? a0 : (a1 !== prv && a1 >= 0 ? a1 : -1);
      if (nx < 0 || (deg[c] === 1 && prv >= 0)) break;
      prv = c;
      c = nx;
    }
    removeEnd(c);
    if (ends.length === 0) break;
    let best = -1;
    let bestD = Infinity;
    for (let k = 0; k < ends.length; k++) {
      const o = ends[k];
      const dx = X[c] - X[o];
      const dy = Y[c] - Y[o];
      const d2 = dx * dx + dy * dy;
      if (d2 < bestD || (d2 === bestD && o < best)) { bestD = d2; best = o; }
    }
    cur = best;
  }
  if (t !== n) throw new Error('greedy tour incomplete');
}

function hasNeighbour(i, j) {
  for (let k = 0; k < K; k++) if (neigh[i * K + k] === j) return true;
  return false;
}

function kick() {
  const maxL = Math.max(1, Math.min(KICK_LEN, Math.floor(n / 8)));
  const r = rand() % n;
  const l1 = 1 + (rand() % maxL);
  const l2 = 1 + (rand() % maxL);
  const i = pos[r];
  const at = (o) => tour[(i + o) % n];
  const a = at(0);
  const b1 = at(1);
  const b2 = at(l1);
  const c1 = at(l1 + 1);
  const c2 = at(l1 + l2);
  const d = at(l1 + l2 + 1);
  const delta = dist(a, c1) + dist(c2, b1) + dist(b2, d) - dist(a, b1) - dist(b2, c1) - dist(c2, d);
  reverse((i + 1) % n, (i + l1 + l2) % n);
  reverse((i + 1) % n, (i + l2) % n);
  reverse((i + l2 + 1) % n, (i + l1 + l2) % n);
  push(a); push(b1); push(b2); push(c1); push(c2); push(d);
  return delta;
}

function solve(text, seed) {
  const lines = text.split('\n');
  n = parseInt(lines[0], 10);
  X = new Float64Array(n);
  Y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const ln = lines[i + 1];
    const sp = ln.indexOf(' ');
    X[i] = parseInt(ln.slice(0, sp), 10);
    Y[i] = parseInt(ln.slice(sp + 1), 10);
  }
  if (n < 8) {
    tour = new Int32Array(n);
    for (let i = 0; i < n; i++) tour[i] = i;
    return tour;
  }
  buildNeighbours();
  greedyTour();
  pos = new Int32Array(n);
  for (let i = 0; i < n; i++) pos[tour[i]] = i;
  queue = new Int32Array(n);
  inQ = new Uint8Array(n);
  qHead = 0; qTail = 0; qCount = 0;
  for (let i = 0; i < n; i++) push(tour[i]);
  journaling = true;
  journal.length = 0;
  localSearch();

  rngState = seed >>> 0 || 1;
  const kicks = Math.round(n * KICKS_PER_CITY);
  for (let it = 0; it < kicks; it++) {
    journal.length = 0;
    const delta = kick() + localSearch();
    if (delta > 0) undoTo(0);
  }
  journal.length = 0;
  return tour;
}

const files = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.txt')).sort();
for (let f = 0; f < files.length; f++) {
  const id = files[f].slice(0, -4);
  const text = readFileSync(join(INPUT_DIR, files[f]), 'utf8');
  const t = solve(text, 0x9e3779b9 ^ (f * 2654435761));
  writeFileSync(join(OUTPUT_DIR, id + '.tour'), Array.from(t).join('\n') + '\n');
}
