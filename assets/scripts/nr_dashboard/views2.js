/* Tabs 4-6: model comparison, acceleration, clip explorer. Plus the app shell. */
'use strict';

/* ============================ TAB 4: models ============================ */

function renderModels() {
  const key = state.metric, st = COL[key].status;
  // The hard refusal: a column that failed the GT-is-ceiling audit cannot rank models,
  // and a paired-only column has no readable absolute level to rank BY.
  if (!CAN_RANK_MODELS(st)) {
    setHTML('#models-body', blocked(key, 'The model ranking') +
      (IS_RESULT(st) ? `<div class="rule" style="margin-top:9px">Its
         <b>acceleration-vs-dense</b> deltas remain claimable \u2014 see the Acceleration tab,
         where the bias cancels inside each pair of generated clips.</div>` : ''));
    return;
  }
  const models = ordered([...new Set(FILTERED.map(c => c.model).filter(m => m && m !== 'gt'))], L.model_order);
  const methods = ordered([...new Set(FILTERED.map(c => c.method).filter(Boolean))], L.method_order);
  const cells = models.map(m => methods.map(me => {
    const v = vals(key, c => c.model === m && c.method === me);
    const md = median(v);
    const claim = v.length >= MIN_N;
    return {
      v: claim ? md : null, n: v.length,
      text: !v.length ? '\u2014' : (claim ? fmt(key, md) : `?(${v.length})`),
      blocked: !claim,
      tip: `<b>${lbl('model', m)} \u00b7 ${lbl('method', me)}</b><br>${esc(COL[key].label)}<br>
        ${claim ? 'median ' + fmt(key, md) + ' ' + esc(COL[key].unit) : `only ${v.length} valid clips \u2014 below the ${MIN_N}-clip floor, no number is claimed`}
        <br>n=${v.length}`,
    };
  }));
  const wrap = $('#models-body');
  wrap.innerHTML = `<div class="sub">Median ${esc(COL[key].label)} ${chip(key)} per model and method,
     over the clips in scope. Cells with fewer than ${MIN_N} valid clips show <span class="mono">?(n)</span>
     instead of a number.</div>`;
  wrap.appendChild(heat(cells, models.map(m => lbl('model', m)), methods.map(m => lbl('method', m)),
    { cw: 74, ch: 22, padL: 116, padT: 44 }));

  // The Real-Video anchor, per embodiment: the honest reference for "how far from real".
  const embs = ordered([...new Set(FILTERED.map(c => c.embodiment))], L.emb_order);
  const anchor = embs.map(e => {
    const gt = vals(key, c => c.embodiment === e && c.method === 'gt');
    const dn = vals(key, c => c.embodiment === e && c.method === 'dense');
    return `<tr><td class="l">${esc(lbl('embodiment', e))}</td>
      <td>${gt.length >= MIN_N ? fmt(key, median(gt)) : '<span class="noclaim">n/a</span>'}</td>
      <td>${gt.length.toLocaleString()}</td>
      <td>${dn.length >= MIN_N ? fmt(key, median(dn)) : '<span class="noclaim">n/a</span>'}</td>
      <td>${dn.length.toLocaleString()}</td></tr>`;
  }).join('');
  wrap.insertAdjacentHTML('beforeend',
    `<h3 style="font-size:12px;margin:16px 0 4px">Real video vs dense, per embodiment</h3>
     <div class="rule" style="margin-bottom:6px">Compare DOWN a row only. Across embodiments the
     quantity itself differs (different camera mounts, different flow field), so a bigger number in
     another row is not a better one.</div>
     <table><thead><tr><th class="l">embodiment</th><th>real video</th><th>n</th><th>dense</th><th>n</th>
     </tr></thead><tbody>${anchor}</tbody></table>`);
}

/* ============================ TAB 5: acceleration ============================ */

function renderAccel() {
  const key = state.metric, st = COL[key].status;
  if (!IS_RESULT(st)) { setHTML('#accel-body', blocked(key, 'Acceleration deltas')); return; }
  const wrap = $('#accel-body');
  wrap.innerHTML = '';
  const note = document.createElement('div');
  note.className = 'sub';
  note.innerHTML = `Each delta is one accelerated clip minus <b>its own dense clip</b> \u2014 same
    episode, same view, same geometry. Pooling across cells is legitimate because a within-clip
    delta contains no absolute level${st === 'CEILING' ? ', which is why this column can carry an acceleration claim even though it cannot rank models' : ''}.`;
  wrap.appendChild(note);

  const forestSlot = document.createElement('div');
  forestSlot.id = 'accel-forest';
  wrap.appendChild(forestSlot);
  renderPooledForest('#accel-forest', key);

  // Per-cell breakdown: is a pooled effect uniform, or driven by one cell?
  const methods = ordered([...new Set(FILTERED.map(c => c.method).filter(m => m && m !== 'gt' && m !== 'dense'))], L.method_order);
  const cellKeys = [];
  const seen = new Set();
  for (const c of FILTERED) {
    const k = `${c.model}|${c.embodiment}|${c.camera_mount}`;
    if (!seen.has(k)) { seen.add(k); cellKeys.push([c.model, c.embodiment, c.camera_mount]); }
  }
  cellKeys.sort((a, b) => (L.emb_order.indexOf(a[1]) - L.emb_order.indexOf(b[1]))
    || (L.model_order.indexOf(a[0]) - L.model_order.indexOf(b[0]))
    || String(a[2]).localeCompare(b[2]));
  // One bucketed pass instead of cells x methods rescans (see core.js deltasBy).
  const buckets = deltasBy(key, c => `${c.model}|${c.embodiment}|${c.camera_mount}|${c.method}`, 'cell');
  const cells = cellKeys.map(([mo, em, mt]) => methods.map(me => {
    const d = buckets.get(`${mo}|${em}|${mt}|${me}`) || [];
    if (!d.length) return { v: null, text: '\u2014', blocked: true, tip: 'no comparable pairs in scope' };
    const md = median(d);
    const claim = d.length >= MIN_N;
    const [lo, hi] = claim ? bootCI(d, 13) : [null, null];
    const p = claim ? wilcoxonP(d) : null;
    const res = resolved(key, d.length, lo, hi, p, md);
    const art = artifactPositive(key, d.length, lo, p);
    return {
      v: md, n: d.length, blocked: !claim,
      text: claim ? fmtSigned(key, md) + (art ? '!' : (res ? '*' : '')) : `?(${d.length})`,
      tip: `<b>${lbl('model', mo)} \u00b7 ${em} \u00b7 ${mt}</b><br>${lbl('method', me)} \u2212 dense<br>
        median ${fmtSigned(key, md)} ${esc(COL[key].unit)}<br>
        ${claim ? `95% CI ${fmt(key, lo)} \u2013 ${fmt(key, hi)}<br>p=${fmtP(p)}<br>` : ''}n=${d.length}
        ${!claim ? `<br><i>below the ${MIN_N}-clip floor: no claim</i>` : (res ? '<br><b>resolved</b>' : '<br>not distinguishable from dense')}
        ${art ? '<br><b style="color:#e06c6c">positive = over-smoothing artifact</b>' : ''}`,
    };
  }));
  wrap.insertAdjacentHTML('beforeend',
    `<h3 style="font-size:12px;margin:18px 0 4px">Per-cell breakdown ${chip(key)}</h3>
     <div class="rule" style="margin-bottom:6px"><span class="mono">*</span> resolved claim &middot;
     <span class="mono">!</span> statistically solid but the direction is the artifact &middot;
     <span class="mono">?(n)</span> fewer than ${MIN_N} pairs, so no claim at any effect size.
     Faded cells carry no claim.</div>`);
  wrap.appendChild(heat(cells,
    cellKeys.map(([mo, em, mt]) => `${lbl('model', mo)} / ${em.slice(0, 4)} / ${mt}`),
    methods.map(m => lbl('method', m)),
    { cw: 72, ch: 20, padL: 176, padT: 44, ramp: ['#2b5d8a', '#8a8a8a', '#b8860b'] }));

  const cav = (POLICY.accel_caveats || {})[key];
  if (cav) wrap.insertAdjacentHTML('beforeend', `<div class="blocked" style="margin-top:12px">
    <b>Caveat on this column's acceleration delta.</b><br>
    <span class="mono" style="font-size:10px">${esc(cav)}</span></div>`);
}

/* ============================ TAB 6: clip explorer ============================ */

function renderClips() {
  const groups = DATA.exemplars;
  const sel = $('#clip-select');
  if (!sel.dataset.filled) {
    sel.innerHTML = groups.map((g, i) =>
      `<option value="${i}">${esc(lbl('model', g.model))} \u00b7 ${esc(g.cell[0])} \u00b7 ${esc(g.cell[1])} \u00b7 ${esc(g.split)} \u00b7 ${esc(String(g.pair_key).split('/').pop())}</option>`).join('');
    sel.dataset.filled = '1';
    sel.addEventListener('change', renderClips);
  }
  const g = groups[+sel.value || 0];
  if (!g) { setHTML('#clip-body', '<div class="blocked">This cache has no exemplar clips with per-frame traces.</div>'); return; }
  const dense = DATA.perframe[g.dense_id];
  const accels = g.accel.map(a => ({ ...a, pf: DATA.perframe[a.id] })).filter(a => a.pf);
  if (!dense) { setHTML('#clip-body', '<div class="blocked">Per-frame trace missing for this group.</div>'); return; }
  const chosen = accels[0];
  const wrap = $('#clip-body');
  wrap.innerHTML = `<div class="sub">${esc(g.instruction || '')}</div>`;

  // The gated values of every metric, dense vs each accel clip on the SAME episode.
  const head = ['metric', 'dense', ...accels.map(a => lbl('method', a.method))];
  const body = KEYS.map(k => {
    const d = (dense.gated || {})[k];
    const tds = accels.map(a => {
      const v = (a.pf.gated || {})[k];
      const dd = (d === null || d === undefined || v === null || v === undefined) ? null : v - d;
      return `<td>${v === null || v === undefined ? '<span class="noclaim">gated out</span>' : fmt(k, v)}
        ${dd === null ? '' : `<span style="color:var(--text-dim)"> (${fmtSigned(k, dd)})</span>`}</td>`;
    }).join('');
    return `<tr><td class="l">${esc(COL[k].label)} ${chip(k)}</td>
      <td>${d === null || d === undefined ? '<span class="noclaim">gated out</span>' : fmt(k, d)}</td>${tds}</tr>`;
  }).join('');
  wrap.insertAdjacentHTML('beforeend',
    `<table style="max-width:820px"><thead><tr>${head.map((h, i) => `<th class="${i ? '' : 'l'}">${esc(h)}</th>`).join('')}</tr></thead>
     <tbody>${body}</tbody></table>
     <div class="rule">One clip. "gated out" means this clip does not count toward that column at all \u2014
     the reason is in the Validity tab.</div>`);

  const series = (field, logy, hline, hlabel, ylabel) => {
    const list = [{ label: 'dense', values: dense[field], color: '#9aa7b5' }]
      .concat(accels.map(a => ({ label: lbl('method', a.method), values: a.pf[field], color: famColor(a.method) })));
    const box = document.createElement('div');
    box.className = 'card';
    box.innerHTML = `<h3>${esc(ylabel)}</h3>
      <div class="legend">${list.map(s => `<span><i style="background:${s.color}"></i>${esc(s.label)}</span>`).join('')}</div>`;
    box.appendChild(lines(series_(list), { w: 700, h: 165, logy, hline, hlabel, ylabel: '' }));
    return box;
  };
  const series_ = l => l;
  const grid = document.createElement('div');
  grid.className = 'grid2';
  grid.appendChild(series('fg_flow_p95', false, POLICY.gates.active_px,
    `active threshold ${POLICY.gates.active_px} px`, 'Foreground p95 flow \u2014 the quantity Actor Activity medians'));
  grid.appendChild(series('fg_flow_ego_p95', false, POLICY.gates.active_px,
    `active threshold ${POLICY.gates.active_px} px`, 'Foreground p95, ego motion removed'));
  grid.appendChild(series('copy_mse', true, POLICY.gates.copy_err_floor,
    `weight-0 floor ${POLICY.gates.copy_err_floor}`, 'copy_mse \u2014 how much the frame moves'));
  grid.appendChild(series('vfi_mse', true, null, '', 'vfi_mse \u2014 how unpredictable the middle frame is'));
  grid.appendChild(series('gain', false, null, '', 'gain \u2014 Active Motion Continuity, per triple'));
  grid.appendChild(series('fg_cov_pct', false, POLICY.gates.min_bg_cov_pct,
    `head bg floor ${POLICY.gates.min_bg_cov_pct}%`, 'Foreground mask coverage (%)'));
  wrap.appendChild(grid);
  wrap.insertAdjacentHTML('beforeend',
    `<div class="rule">A pruning method usually lowers Actor Activity by flattening the PEAKS, not by
     making every pair still \u2014 and it can raise Active Motion Continuity simply by lowering
     <span class="mono">copy_mse</span> (less motion), which is the artifact, not an improvement.
     ${chosen ? '' : ''}</div>`);
}

/* ============================ app shell ============================ */

const TABS = [
  ['overview', 'Overview', renderOverview],
  ['validity', 'Validity &amp; coverage', renderValidity],
  ['redundancy', 'Redundancy', renderRedundancy],
  ['models', 'Models', renderModels],
  ['accel', 'Acceleration', renderAccel],
  ['clips', 'Clip explorer', renderClips],
];

function buildFilters() {
  const host = $('#filters');
  const nice = { embodiment: 'Embodiment', model: 'Model', split: 'Split', camera_mount: 'Camera mount', method: 'Method' };
  host.innerHTML = FIELDS.map(f => `<div class="fgroup"><label>${nice[f]}</label>
      <div class="pills" data-field="${f}">
        ${state.domain[f].map(v => `<span class="pill" role="button" data-v="${esc(v)}" aria-pressed="false">${esc(lbl(f, v))}</span>`).join('')}
      </div></div>`).join('')
    + `<div class="fgroup"><label>Metric in focus</label>
        <select id="metric-select">${DATA.columns.map(c =>
      `<option value="${c.key}"${c.key === state.metric ? ' selected' : ''}>${esc(c.label)} \u2014 ${SS[COL[c.key].status].short}</option>`).join('')}</select></div>`
    + `<div class="fgroup"><label>&nbsp;</label><button id="clear-filters">Clear filters</button></div>`
    + `<div id="matchcount"></div>`;
  host.querySelectorAll('.pill').forEach(p => p.addEventListener('click', () => {
    const f = p.parentElement.dataset.field, v = p.dataset.v;
    const s = state.sel[f];
    s.has(v) ? s.delete(v) : s.add(v);
    p.setAttribute('aria-pressed', s.has(v) ? 'true' : 'false');
    apply();
  }));
  $('#metric-select').addEventListener('change', e => { state.metric = e.target.value; apply(); });
  $('#clear-filters').addEventListener('click', () => {
    FIELDS.forEach(f => state.sel[f].clear());
    host.querySelectorAll('.pill').forEach(p => p.setAttribute('aria-pressed', 'false'));
    apply();
  });
}

function buildTabs() {
  const nav = $('#tabs');
  nav.innerHTML = TABS.map(([id, label]) =>
    `<button role="tab" data-tab="${id}" aria-selected="${id === state.tab}">${label}</button>`).join('');
  nav.querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
    state.tab = b.dataset.tab;
    nav.querySelectorAll('button').forEach(x => x.setAttribute('aria-selected', String(x === b)));
    apply();
  }));
}

/* Recompute only the visible tab: a full six-tab rebuild on every pill click is ~1s at
 * 39k clips, one tab is ~60ms. */
function apply() {
  refilter();
  $('#matchcount').textContent = `${FILTERED.length.toLocaleString()} of ${CLIPS.length.toLocaleString()} clips in scope`;
  TABS.forEach(([id]) => $('#tab-' + id).classList.toggle('hidden', id !== state.tab));
  const t0 = performance.now();
  const fn = (TABS.find(t => t[0] === state.tab) || [])[2];
  if (fn) fn();
  $('#rendertime').textContent = `${(performance.now() - t0).toFixed(0)} ms`;
}

function initTheme() {
  let t = null;
  try { t = localStorage.getItem('nr-dash-theme'); } catch (_) { }
  if (t !== 'light' && t !== 'dark') t = matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', t);
  $('#theme-toggle').addEventListener('click', () => {
    const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('nr-dash-theme', next); } catch (_) { }
    apply();
  });
}

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  buildFilters();
  buildTabs();
  apply();
});
