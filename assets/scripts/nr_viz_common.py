"""Shared plumbing for the NR metric figures: cache loading, policy badges, styling.

Split out of `make_nr_metric_figures.py` so the dashboard builder can reuse the exact
same status/badge/colour logic -- a metric must not be labelled VALID in a figure and
DROPPED in the dashboard.
"""
import gzip
import json
from pathlib import Path

# Status -> (short badge, long caption fragment, colour). The colours are the only place
# the "how much may I trust this column" signal lives, so they are deliberately loud.
STATUS_STYLE = {
    'VALID': ('VALID', 'readable as a level and as a delta', '#2f7d4f'),
    'DESCRIPTIVE': ('DESCRIPTIVE', 'no polarity -- more is neither better nor worse',
                    '#4a7fb5'),
    'CEILING': ('CEILING-FAILED', 'descriptive only -- no model ranking; the '
                'accel-vs-dense delta IS claimable', '#b8860b'),
    'PAIRED_ONLY': ('PAIRED-ONLY', 'absolute medians mix per-caption bias; read '
                    'paired deltas only', '#1f6f94'),
    'DIAGNOSTIC': ('DIAGNOSTIC', 'computed, never a result', '#6b6b6b'),
    'DROPPED': ('DROPPED', 'invalid metric -- autopsy only', '#a32b2b'),
}
# One-line "what you may / may not read" per status, printed on every panel. Short on
# purpose: a caption nobody reads protects nobody.
STATUS_RULE = {
    'VALID': 'OK: level + delta',
    'DESCRIPTIVE': 'OK: level + delta.  NOT: "higher = better"',
    'CEILING': 'OK: delta vs dense.  NOT: model ranking',
    'PAIRED_ONLY': 'OK: paired delta.  NOT: absolute median',
    'DIAGNOSTIC': 'OK: validity audit.  NOT: any result',
    'DROPPED': 'NOT a result in any form (see F12)',
}
METHOD_ORDER = ['gt', 'dense', 'itm', 'sito', 'worldcache', 'fastercache', 'dicache',
                'svg2', 'svg1', 'radial', 'pisa']
METHOD_LABEL = {'gt': 'Real Video', 'dense': 'Dense', 'itm': 'ITM', 'sito': 'SiTo',
                'worldcache': 'World-cache', 'fastercache': 'Faster-cache',
                'dicache': 'Di-cache', 'svg2': 'SVG2', 'svg1': 'SVG1',
                'radial': 'Radial', 'pisa': 'PISA'}
MODEL_LABEL = {'dreamgen': 'DreamGen', 'dreamdojo': 'DreamDojo',
               'ctrlworld': 'Ctrl-World', 'ctrlworld_4view_grid': 'CW-4view'}
MODEL_ORDER = ['ctrlworld', 'ctrlworld_4view_grid', 'dreamdojo', 'dreamgen']
EMB_ORDER = ['bimanual', 'humanoid', 'single_arm']
EMB_LABEL = {'bimanual': 'Bimanual (ALOHA)', 'humanoid': 'Humanoid (GR1)',
             'single_arm': 'Single-arm (DROID)'}
SPLIT_LABEL = {'makovian': 'Markovian', 'non_makovian': 'Non-Markovian'}
MOUNT_ORDER = ['static', 'head', 'wrist']
# Method families, used to colour forests and the Pareto plot consistently.
FAMILY = {'itm': 'pruning', 'sito': 'pruning',
          'worldcache': 'caching', 'fastercache': 'caching', 'dicache': 'caching',
          'svg1': 'sparse-attn', 'svg2': 'sparse-attn', 'radial': 'sparse-attn',
          'pisa': 'sparse-attn'}
FAMILY_COLOR = {'pruning': '#c4622d', 'caching': '#2f7d4f', 'sparse-attn': '#3b5fa8',
                'dense': '#333333', 'gt': '#7a7a7a'}


class Cache:
    """The whole cache, loaded once. `clips` is a list of slim dicts; `agg` and `meta`
    are the JSON blobs `nr_viz_cache.py` wrote."""

    def __init__(self, cache_dir, load_clips=True):
        self.dir = Path(cache_dir)
        self.agg = json.loads((self.dir / 'agg.json').read_text())
        self.meta = json.loads((self.dir / 'meta.json').read_text())
        self.columns = self.agg['columns']
        self.policy = self.agg['policy']
        self.keys = [c['key'] for c in self.columns]
        self.col = {c['key']: c for c in self.columns}
        self.clips = []
        if load_clips:
            with gzip.open(self.dir / 'clips.jsonl.gz', 'rt') as fh:
                self.clips = [json.loads(line) for line in fh if line.strip()]

    # --- metric metadata -------------------------------------------------------
    def label(self, key):
        return self.col[key]['label']

    def unit(self, key):
        return self.col[key]['unit']

    def status(self, key):
        """Policy status, refined by polarity.

        The cache writes VALID for anything no policy dict touches, but `actor_motion_raw`
        has `higher=None` -- it is a descriptive quantity where "more" is neither good
        nor bad. Labelling it VALID next to a True-polarity column invites exactly the
        "Actor Activity went up, so the method is better" reading that `T.COLUMNS`
        annotates against.
        """
        st = self.col[key]['status']
        if st == 'VALID' and self.col[key]['higher'] is None:
            return 'DESCRIPTIVE'
        return st

    def digits(self, key):
        return self.col[key]['digits']

    def polarity(self, key):
        """True = higher better, False = lower better, None = descriptive."""
        return self.col[key]['higher']

    def polarity_arrow(self, key):
        h = self.polarity(key)
        return 'higher is better' if h else ('lower is better' if h is False
                                             else 'descriptive (no polarity)')

    def why(self, key):
        """The measured reason behind a non-VALID status, verbatim from the policy
        dicts, so a caption can never paraphrase it into something weaker."""
        for name in ('dropped', 'not_reported', 'paired_only', 'ceiling_failed',
                     'not_cross_embodiment'):
            d = self.policy.get(name) or {}
            if key in d:
                return d[key]
        return None

    # --- per-clip access -------------------------------------------------------
    def values(self, key, clips=None, **filters):
        """Gated values of `key` over the clips matching `filters` (exact match on any
        strata field; a list/tuple value means 'in')."""
        return [v for v in (c['gated'].get(key)
                            for c in self.select(clips, **filters)) if v is not None]

    def select(self, clips=None, **filters):
        out = self.clips if clips is None else clips
        for field, want in filters.items():
            if want is None:
                continue
            if isinstance(want, (list, tuple, set, frozenset)):
                out = [c for c in out if c.get(field) in want]
            else:
                out = [c for c in out if c.get(field) == want]
        return out

    def exemplar_groups(self):
        return self.agg.get('exemplars') or []

    def perframe(self, clip_id):
        p = self.dir / 'perframe' / f'{clip_id}.json'
        return json.loads(p.read_text()) if p.exists() else None

    # --- provenance ------------------------------------------------------------
    @property
    def partial(self):
        return bool(self.meta.get('partial'))

    def stamp(self):
        n, tot = self.meta.get('scored_clips'), self.meta.get('manifest_clips')
        frac = f' ({100.0 * n / tot:.0f}%)' if tot else ''
        state = 'PARTIAL' if self.partial else 'COMPLETE'
        return f'{state} {n}/{tot}{frac} view-clips | protocol {self.meta.get("protocol")}'


def method_label(m):
    return METHOD_LABEL.get(m, m or '?')


def model_label(m):
    return MODEL_LABEL.get(m, m)


def sort_methods(methods):
    return sorted(methods, key=lambda m: (METHOD_ORDER.index(m)
                                          if m in METHOD_ORDER else 99, m))


def sort_models(models):
    return sorted(models, key=lambda m: (MODEL_ORDER.index(m)
                                         if m in MODEL_ORDER else 99, m))


def family_color(method):
    if method in ('dense', 'gt'):
        return FAMILY_COLOR[method]
    return FAMILY_COLOR.get(FAMILY.get(method), '#888888')


def wrap(text, width=110):
    """Greedy wrap; captions carry verbatim policy strings that are 300+ chars."""
    words, lines, cur = (text or '').split(), [], ''
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f'{cur} {w}'.strip()
    if cur:
        lines.append(cur)
    return '\n'.join(lines)
