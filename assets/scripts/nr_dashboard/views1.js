/* Tab renderers. Each `render*` writes into its panel from FILTERED, so every tab
 * responds to the global filter with re-medianed numbers. */
'use strict';

const $ = sel => document.querySelector(sel);
function setHTML(sel, html) { const n = $(sel); if (n) n.innerHTML = html; }
function mount(sel, node) { const n = $(sel); if (n) { n.innerHTML = ''; n.appendChild(node); } }

/* A reusable refusal block: the UI states WHY a view is withheld rather than
 * silently omitting it, otherwise a reader concludes the data is missing. */
function blocked(key, what) {
  const c = COL[key], s = SS[c.status];
  return `<div class="blocked">${chip(key)} <b>${what} is not shown for ${esc(c.label)}.</b><br>
    ${esc(s.caption)}${c.why ? '<br><br><span class="mono" style="font-size:10px">' + esc(c.why) + '</span>' : ''}</div>`;
}

/* ============================ TAB 1: overview ============================ */

function renderOverview() {
  const cards = DATA.columns.map(col => {
    const key = col.key, st = COL[key].status;
    const mounts = state.domain.camera_mount.filter(m => !state.sel.camera_mount.size || state.sel.camera_mount.has(m));
    // One bucketed pass per metric instead of one scan per mount; memoized so that
    // switching tabs or themes does not redo it.
    const groups = memo(`mountvals:${key}`, () => {
      const i = COL[key].idx, by = new Map();
      for (const c of FILTERED) {
        const v = c.v[i];
        if (v === null || v === undefined) continue;
        let a = by.get(c.camera_mount);
        if (!a) { a = []; by.set(c.camera_mount, a); }
        a.push(v);
      }
      return mounts.map(m => ({ label: m, color: SS[st].color, values: by.get(m) || [] }))
        .filter(g => g.values.length);
    });
    const all = memo(`vals:${key}`, () => vals(key));
    const md = median(all);
    const ci = all.length >= 3 ? bootCI(all, 7) : [null, null];
    const holder = document.createElement('div');
    holder.className = 'card' + (IS_RESULT(st) ? '' : ' muted');
    holder.innerHTML = `<h3>${esc(col.label)} ${chip(key)}</h3>
      <div class="unit">${esc(col.unit)} &middot; ${esc(col.polarity_text)}</div>
      <div style="font-size:19px;font-weight:650;line-height:1.1">
        ${CAN_SHOW_ABSOLUTE(st) ? fmt(key, md) : '<span class="noclaim" style="font-size:12px">absolute median withheld</span>'}
      </div>
      <div style="font-size:10.5px;color:var(--text-muted)">
        ${CAN_SHOW_ABSOLUTE(st) && ci[0] !== null ? `95% CI ${fmt(key, ci[0])} \u2013 ${fmt(key, ci[1])} &middot; ` : ''}n=${all.length.toLocaleString()}
      </div>
      <div class="rule">${esc(col.rule)}</div>
      ${col.why ? `<div class="why" onclick="this.classList.toggle('open')">${esc(col.why)}</div>` : ''}
      <div class="chartslot"></div>`;
    if (CAN_SHOW_ABSOLUTE(st) && groups.length) {
      holder.querySelector('.chartslot').appendChild(
        boxes(groups, { w: 420, h: 168, padL: 46, padB: 40 }));
    } else if (!CAN_SHOW_ABSOLUTE(st)) {
      holder.querySelector('.chartslot').innerHTML =
        `<div class="rule" style="margin-top:9px">Per-mount distribution withheld: this column's
         absolute level carries a per-caption bias, so only paired deltas (Acceleration tab) are readable.</div>`;
    }
    return holder;
  });
  const wrap = $('#overview-cards');
  wrap.innerHTML = '';
  cards.forEach(c => wrap.appendChild(c));

  // Headline: the accel deltas for the currently selected metric, re-medianed.
  renderPooledForest('#overview-forest', state.metric);
  setHTML('#overview-forest-note',
    `Median paired delta (accel \u2212 its own dense clip) for <b>${esc(COL[state.metric].label)}</b>,
     recomputed over the ${FILTERED.length.toLocaleString()} clips currently in scope.
     Intervals are a ${BOOT_B}-resample percentile bootstrap, or an exact order-statistic
     interval once a cell exceeds ${BOOT_EXACT_ABOVE.toLocaleString()} pairs; the published
     tables always use a ${POLICY.bootstrap_resamples.toLocaleString()}-resample bootstrap.`);
}

/* Pooled accel forest for one metric, computed client-side from FILTERED. */
function renderPooledForest(sel, key) {
  const st = COL[key].status;
  if (!IS_RESULT(st)) { setHTML(sel, blocked(key, 'The acceleration forest')); return; }
  const methods = ordered(
    [...new Set(FILTERED.map(c => c.method).filter(m => m && m !== 'gt' && m !== 'dense'))],
    L.method_order);
  if (!methods.length) { setHTML(sel, '<div class="blocked">No acceleration method is in scope. Widen the Method filter.</div>'); return; }
  const buckets = deltasBy(key, c => c.method, 'method');
  const rows = methods.map(m => {
    const d = buckets.get(m) || [];
    const md = median(d);
    const [lo, hi] = d.length >= 3 ? bootCI(d, 11) : [null, null];
    const p = wilcoxonP(d);
    const art = artifactPositive(key, d.length, lo, p);
    return {
      label: lbl('method', m), n: d.length, med: md, lo, hi, p,
      resolved: resolved(key, d.length, lo, hi, p, md), artifact: art,
      color: famColor(m),
      tip: `<b>${lbl('method', m)}</b> \u2212 dense<br>${esc(COL[key].label)}<br>
        median ${fmtSigned(key, md)} ${esc(COL[key].unit)}<br>
        95% CI ${fmt(key, lo)} \u2013 ${fmt(key, hi)} <span style="color:var(--text-dim)">(${ciKind(d.length)})</span><br>
        n=${d.length} paired clips, p=${fmtP(p)}
        ${art ? '<br><b style="color:#e06c6c">positive here is the over-smoothing artifact, not a win</b>' : ''}
        ${d.length < MIN_N ? `<br><i>n below ${MIN_N}: no claim</i>` : ''}`,
    };
  });
  const node = forest(rows, { w: 620, unit: `delta vs dense (${COL[key].unit})` });
  const holder = document.createElement('div');
  holder.appendChild(node);
  const cs = DATA.common_support[key];
  if (cs && cs.cells) {
    const moved = Object.entries(cs.moved || {}).filter(([, v]) => v);
    holder.insertAdjacentHTML('beforeend',
      `<div class="rule">Published common-support check: ${cs.cells} cells are covered by all
       ${cs.methods.length} methods${moved.length ? '; ranking there moves ' +
        moved.slice(0, 4).map(([m, v]) => `${lbl('method', m)} ${v > 0 ? '+' : ''}${v}`).join(', ') : ''}.</div>`);
  }
  const n = $(sel); n.innerHTML = ''; n.appendChild(holder);
}

/* ============================ TAB 2: validity ============================ */

function renderValidity() {
  const rows = DATA.columns.map(col => {
    const key = col.key, i = COL[key].idx;
    let ok = 0; const why = new Map();
    for (const c of FILTERED) {
      if (c.v[i] !== null && c.v[i] !== undefined) ok++;
      else { const r = c.reasonOf(i) || 'not computed for this clip'; why.set(r, (why.get(r) || 0) + 1); }
    }
    const top = [...why.entries()].sort((a, b) => b[1] - a[1]);
    const total = FILTERED.length || 1;
    return `<tr>
      <td class="l">${esc(col.label)} ${chip(key)}</td>
      <td>${ok.toLocaleString()}</td>
      <td>${(100 * ok / total).toFixed(1)}%</td>
      <td>${(total - ok).toLocaleString()}</td>
      <td class="l" style="white-space:normal;font-size:10.5px;color:var(--text-muted)">
        ${top.slice(0, 3).map(([r, n]) => `${esc(r)} <b>${n}</b>`).join(' &middot; ') || '\u2014'}</td>
    </tr>`;
  }).join('');
  setHTML('#validity-table', `<table><thead><tr>
      <th class="l">metric</th><th>valid clips</th><th>of scope</th><th>gated out</th>
      <th class="l">top gate reasons in this scope</th></tr></thead><tbody>${rows}</tbody></table>`);

  const g = POLICY.gates;
  setHTML('#validity-gates', Object.entries({
    'tracked pair fraction': `\u2265 ${g.min_tracked_frac}`,
    'AMC weight sum': `\u2265 ${g.min_weight_sum}`,
    'active pairs': `\u2265 ${g.min_active_pairs} and \u2265 ${g.min_active_frac} of pairs`,
    'head-mount background coverage': `\u2265 ${g.min_bg_cov_pct}%`,
    'active threshold': `${g.active_px} px / source frame`,
    'copy-error floor (zero weight below)': `${g.copy_err_floor}`,
    'max clip-length mismatch when pairing': `${g.max_length_mismatch}`,
    'min clips for any claim': `${POLICY.min_pairs_for_claim}`,
    'significance level': `${POLICY.alpha}`,
  }).map(([k, v]) => `<tr><td class="l">${esc(k)}</td><td class="mono">${esc(v)}</td></tr>`).join(''));

  const cov = DATA.coverage.stages;
  setHTML('#validity-pipeline', Object.entries({
    'clips in manifest (expected)': cov.manifest_clips,
    'result files written': cov.result_files,
    'wrong protocol (excluded)': cov.wrong_protocol,
    'unreadable (excluded)': cov.unreadable,
    'in protocol (the cache)': cov.in_protocol,
    'currently in scope (filters)': FILTERED.length,
  }).map(([k, v]) => `<tr><td class="l">${esc(k)}</td><td>${(v || 0).toLocaleString()}</td></tr>`).join(''));
}

/* ============================ TAB 3: redundancy ============================ */

function renderRedundancy() {
  const mounts = state.domain.camera_mount.filter(m => !state.sel.camera_mount.size || state.sel.camera_mount.has(m));
  const scopes = [['all mounts in scope', null], ...mounts.map(m => [`mount = ${m}`, m])];
  // The whole grid is one memo entry: it is the most expensive view (a rank sort per
  // metric pair per scope) and nothing in it depends on the selected metric.
  const computed = memo('redundancy', () => scopes.map(([name, mnt]) => {
    const subset = mnt ? FILTERED.filter(c => c.camera_mount === mnt) : FILTERED;
    if (subset.length < 30) return null;
    const grid = KEYS.map(a => KEYS.map(b => a === b ? { rho: 1, n: subset.length } : spearmanJS(subset, a, b)));
    return { name, n: subset.length, grid };
  }).filter(Boolean));

  const wrap = $('#redundancy-grids');
  wrap.innerHTML = '';
  computed.forEach(({ name, n, grid }) => {
    const cells = KEYS.map((a, i) => KEYS.map((b, j) => {
      const r = grid[i][j];
      if (a === b) return { v: 1, text: '1', tip: `${esc(COL[a].label)} with itself` };
      const strong = r.rho !== null && Math.abs(r.rho) >= 0.7;
      return {
        v: r.rho, n: r.n,
        text: r.rho === null ? 'n/a' : (r.rho > 0 ? '+' : '') + r.rho.toFixed(2),
        tip: `<b>${esc(COL[a].label)}</b> vs <b>${esc(COL[b].label)}</b><br>
              Spearman rho ${r.rho === null ? 'n/a' : r.rho.toFixed(3)} over n=${r.n} clips
              ${strong ? '<br><b>|rho| \u2265 0.7: these two are largely the same measurement</b>' : ''}`,
      };
    }));
    const short = KEYS.map(k => COL[k].label.replace('Active Motion Continuity', 'AMC'));
    const box = document.createElement('div');
    box.className = 'card';
    box.innerHTML = `<h3>${esc(name)}</h3><div class="unit">n=${n.toLocaleString()} clips in scope</div>`;
    box.appendChild(heat(cells, short, short, { cw: 58, ch: 20, padL: 132, padT: 46, ramp: ['#2b5d8a', '#8a8a8a', '#a83232'] }));
    wrap.appendChild(box);
  });
  if (!wrap.children.length) wrap.innerHTML = '<div class="blocked">Fewer than 30 clips in scope; correlations are not computed.</div>';

  // Ranked verdicts, reusing the pooled grid computed above.
  const pooled = computed.find(c => c.name.startsWith('all'));
  const pairs = [];
  if (pooled) {
    for (let i = 0; i < KEYS.length; i++) {
      for (let j = i + 1; j < KEYS.length; j++) {
        const r = pooled.grid[i][j];
        if (r.rho !== null) pairs.push([KEYS[i], KEYS[j], r.rho, r.n]);
      }
    }
  }
  pairs.sort((a, b) => Math.abs(b[2]) - Math.abs(a[2]));
  setHTML('#redundancy-table', `<table><thead><tr><th class="l">metric A</th><th class="l">metric B</th>
    <th>rho</th><th>n</th><th class="l">reading</th></tr></thead><tbody>` +
    pairs.map(([a, b, rho, n]) => `<tr>
      <td class="l">${esc(COL[a].label)} ${chip(a)}</td>
      <td class="l">${esc(COL[b].label)} ${chip(b)}</td>
      <td class="${Math.abs(rho) >= 0.7 ? 'resolved' : ''}">${(rho > 0 ? '+' : '') + rho.toFixed(3)}</td>
      <td>${n.toLocaleString()}</td>
      <td class="l" style="color:var(--text-muted)">${Math.abs(rho) >= 0.7 ? 'redundant \u2014 reporting both double-counts one signal'
      : Math.abs(rho) >= 0.35 ? 'shares a substantial component' : 'largely independent'}</td>
    </tr>`).join('') + '</tbody></table>');
}

function spearmanJS(subset, a, b) {
  const ia = COL[a].idx, ib = COL[b].idx;
  const xs = [], ys = [];
  for (const c of subset) {
    const x = c.v[ia], y = c.v[ib];
    if (x === null || x === undefined || y === null || y === undefined) continue;
    xs.push(x); ys.push(y);
  }
  const n = xs.length;
  if (n < 8) return { rho: null, n };
  const rx = ranks(xs), ry = ranks(ys);
  const mx = (n + 1) / 2;
  let num = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) { num += (rx[i] - mx) * (ry[i] - mx); dx += (rx[i] - mx) ** 2; dy += (ry[i] - mx) ** 2; }
  return { rho: (dx && dy) ? num / Math.sqrt(dx * dy) : null, n };
}

/* Mid-ranks with tie averaging, matching scipy's default and therefore the Python side. */
function ranks(v) {
  const n = v.length;
  const ord = Array.from({ length: n }, (_, i) => i).sort((p, q) => v[p] - v[q]);
  const r = new Float64Array(n);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && v[ord[j + 1]] === v[ord[i]]) j++;
    const avg = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) r[ord[k]] = avg;
    i = j + 1;
  }
  return r;
}
