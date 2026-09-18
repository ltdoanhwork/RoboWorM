"""Paper tables from a dense NR run: per-stratum medians with bootstrap CI, and the
paired GT-delta that is the headline claim.

Two tables, because they answer different questions and have very different power:

  T1 descriptive : median + 95% bootstrap CI + n + exclusion rate, per
                   (model, embodiment, camera_mount). Never pooled across mounts or
                   fps. Read as "what does this model's output look like", NOT as a
                   ranking -- at n=30 the unpaired CIs of GT vs pred overlap.
  T2 paired      : per pred clip, metric(pred) - metric(gt_<model>) on the SAME
                   episode+view+geometry, median + 95% CI + win rate + Wilcoxon.
                   This is the test with power: measured +1.00 [+0.56,+1.60] for
                   VisualIntegrity where the unpaired comparison could not resolve.

Applies the v3.2 validity gates post-hoc so runs scored by older code are directly
comparable (both gates are recoverable from the stored per-clip record).

Usage:
  python scripts/table_dense_nr.py --run metrics/WorldScore/results/dense_nr_v3_div30
"""
import argparse
import collections
import csv
import json
import math
import random
from pathlib import Path

# Paper columns, in table order. `higher` = direction, None = descriptive only.
# `digits` matters: ActorFlicker lives at ~0.01, so 2 decimals prints every delta
# as "0.00" and hides a resolved effect.
# Columns that FAILED the GT-is-ceiling audit (E1, 2026-09-16) and therefore may
# not be used to rank models. They stay in the table as descriptive statistics
# with the caveat printed, per the v3.2 audit decision.
CEILING_FAILED = {
    'visual_integrity':
        'CLIP-IQA+ fails GT-is-ceiling: pred beats co-located GT on dreamdojo '
        '(+0.90, p=5e-05), 4view (+0.52) and ties ctrlworld (-0.02). Motion, '
        'smoothness and sharpness hypotheses were all rejected. Descriptive only '
        '- do NOT rank models on this column.',
}

# Columns EXCLUDED from the paper by a pre-registered experiment. Still computed by
# run_dense_nr.py (the data stays), but must not appear in a paper table.
DROPPED = {
    'actor_flicker_raw':
        'E7 accel-artifact sanity (2026-09-16): ActorFlicker moves in the WRONG '
        'direction ("improves") for blur at every severity tested, with the 95% CI '
        'excluding 0 - sigma1.6 -0.005 [-0.0069, -0.0026], sigma0.8 -0.0024, and even '
        'at realistic severity (sharpness 0.684x, blurrier than ITM) -0.0010 '
        '[-0.0020, -0.0004]. For frame duplication, temporal smoothing and blocking it '
        'does not react at all (CI spans 0). So it is either wrong or blind on every '
        'artifact tested, and cannot be renormalised into correctness because the '
        'artifact itself suppresses the residual it measures. Also motion-confounded '
        '(rho +0.70 static) with deltas at the ~5e-3 noise floor. '
        'See docs/dense_nr_experiments.md E3 and the E7 section of dense_nr_full_run.md.',
}

# A bootstrap CI of the median at n=3 resamples 3 numbers -- it can "exclude 0" purely
# because all three happen to share a sign, which is 1-in-4 by chance. Two v3.2 rows
# were RESOLVED at n=3 (ctrlworld/humanoid/head, dreamgen/humanoid/wrist). Require a
# floor before any row is allowed to carry a claim; the delta is still printed.
MIN_PAIRS_FOR_CLAIM = 8

# A CI excluding 0 and a Wilcoxon p disagree on skewed delta distributions (measured:
# pisa Visual Integrity, median -0.68, CI [-1.25, -0.27], p=0.262). A claim requires
# BOTH, so p >= ALPHA vetoes the CI. No multiplicity correction is applied -- each
# method x column is a pre-registered comparison, not a search -- so read borderline
# p values with that in mind.
ALPHA = 0.05

# Two clips of a paired delta must be the same measurement. Relative length tolerance
# between an accel clip and its dense counterpart; source_fps and effective_fps must match
# exactly. See comparable_pair().
MAX_LENGTH_MISMATCH = 0.10

# Mounts where a column is known to be confounded, so a resolved row still needs a
# printed caveat. E1 second pass: on head, rho vs actor_motion_raw is -0.541 for AMC
# and -0.553 for VisualIntegrity (both fail the |rho|<0.35 bar, and only on head --
# head is the only mount using ego-translation removal).
CONFOUNDED_MOUNTS = {
    'active_motion_continuity': {'head': 'rho -0.541 vs motion on head mounts'},
    'visual_integrity': {'head': 'rho -0.553 vs motion on head mounts'},
}

# Columns whose ABSOLUTE value is not comparable across embodiments, so only the
# within-embodiment paired delta may be read. Measured 2026-09-16 on the pooled run:
# humanoid GT scores AMC ~48 while bimanual/single_arm GT score ~66, in EVERY motion
# bin (so it is not the motion confound) and at identical analysis geometry -- GR1
# video is intrinsically less interpolable, which lowers the VFI-gain denominator for
# GT and pred alike. Humanoid is consequently the only embodiment where pred "beats"
# GT on AMC (+1.43, pred wins 67%), which is a baseline offset, not model quality.
NOT_CROSS_EMBODIMENT = {
    'active_motion_continuity':
        'humanoid GT AMC ~48 vs ~66 for bimanual/single_arm at matched motion and '
        'identical geometry. Do NOT compare absolute AMC across embodiments, and do '
        'not read a small positive humanoid GT-delta as beating reality.',
}
# Columns that are valid but need an artifact-specific caveat when comparing an
# ACCELERATION method against its dense counterpart (E7).
# Columns whose GT-ceiling failure does NOT carry over to the accel-vs-dense
# comparison, per E7 verdict 4. VisualIntegrity fails GT-is-ceiling because it is
# insensitive to the generative-vs-real gap -- but that bias CANCELS when a method
# clip is differenced against its OWN dense counterpart, and E7 measured VI as the
# most sensitive correct detector of exactly the spatial artifacts accel methods
# produce (blur -6.79, block -12.18, and still resolved at realistic severity
# blur_real -0.93 [-1.23, -0.28]). So a VI accel delta may carry a claim even though
# a VI model ranking may not.
CEILING_FAILURE_CANCELS_VS_DENSE = {'visual_integrity'}

# Columns that are valid but need an artifact-specific caveat when comparing an
# ACCELERATION method against its dense counterpart (E7).
ACCEL_CAVEATS = {
    'active_motion_continuity':
        'AMC is blur-blind: a synthetic blur that cuts sharpness to 0.072x buys '
        '+3.64 AMC [+1.92, +9.06]. At realistic severity (0.684x, worse than ITM 0.741x '
        'or SiTo 0.758x) the bias is only +0.50 [-0.17, +1.00] = NOT significant, so '
        'AMC is usable here. But for the blur-like methods (sito 0.758x, itm 0.741x, '
        'svg2 0.932x sharpness vs dense) a small POSITIVE AMC delta (<= ~+0.5) must not '
        'be read as "better than dense" - report it next to the sharpness ratio from '
        'scripts/_exp_accel_character.py.',
}

# Columns where a POSITIVE accel-vs-dense delta cannot be an improvement, because the
# artifact that produces it is the very thing the metric mis-rewards. Enforced in
# resolved_flag(): a positive delta here never becomes a claim, it becomes an artifact
# flag (see artifact_positive()).
#
# Measured 2026-09-17. The main-results matrix produced 4 cells where an accel method
# beat its own dense counterpart on AMC with the CI excluding 0 and p down to 2e-05.
# comparable_pair() had already cleared them (identical source_fps, effective_fps,
# frame count, analysis geometry and mount), so this is not the render-mismatch bug.
# Measuring the pixels of those exact pairs shows what it is -- second-order temporal
# JERK, i.e. how far the middle frame sits from the linear interpolation of its
# neighbours, which is precisely AMC's VFI-gain denominator:
#
#   cell (n=10 pairs each)                dAMC    sharpness  temporal   JERK
#   fastercache dreamgen/humanoid/MV     +7.02      x0.918    x0.897   x0.859
#   pisa   ctrlworld_4view/humanoid/MV   +3.80      x0.941    x0.913   x0.896
#   svg1   ctrlworld_4view/humanoid/MV   +2.79      x0.970    x0.918   x0.905
#   -- controls, resolved NEGATIVE AMC --
#   itm    dreamgen/humanoid/MV          -4.72      x0.759    x1.376   x1.275
#   sito   dreamgen/bimanual/MV          -9.77      x0.746    x1.001   x0.965
#
# Every positive-AMC cell has JERK < 1 (output MORE predictable than dense) and every
# negative one has JERK >= 0.96. So a positive AMC delta measures over-smoothing, not
# quality. Note this is a LARGER effect (+7.02) than the ACCEL_CAVEATS blur bias
# (+0.50 at realistic sharpness) because it is driven by temporal smoothing, which the
# E7 synthetic blur did not reproduce -- the caveat's "<= ~+0.5" bound is too small.
NO_IMPROVEMENT_VS_DENSE = {
    'active_motion_continuity':
        'A positive AMC delta vs dense is an over-smoothing artifact, not an '
        'improvement: AMC scores how predictable the middle frame of each triple is, '
        'so temporally smoother output scores higher. Measured on the 3 resolved '
        'positive cells: temporal jerk x0.86-x0.91 vs dense, while every resolved '
        'NEGATIVE cell sits at x0.96-x1.28.',
}

# Columns kept but restricted to the paired (pred vs own GT) table, because their
# absolute value carries a bias that only cancels within a pair.
PAIRED_ONLY = {
    'text_condition_alignment':
        'CLIPScore carries a large per-caption style/length bias (~7 points vs ~2 for '
        'image-text agreement), so unpaired medians across different captions are not '
        'comparable. In a pair, pred and its own GT are scored against the SAME prompt '
        '(83/83 pairs verified), so the bias cancels exactly. Report paired deltas only. '
        'The E2 single-shuffle control that originally dropped this column was itself '
        'confounded by that bias; the double-centered matrix control '
        '(scripts/_exp_text_matrix.py) shows CLIPScore does discriminate. '
        'See docs/dense_nr_experiments.md E2.',
}

COLUMNS = [
    # direction forced to None: audit says this cannot express "better" (see CEILING_FAILED)
    ('visual_integrity', 'Visual Integrity', None, 2),
    # Restored 2026-09-16 as a PAIRED-ONLY column. The E2 single-shuffle control read
    # as "saturated" only because CLIPScore carries a per-caption style/length bias
    # (~7 points) that swamps image-text agreement (~2). The clip-by-prompt matrix
    # control (`scripts/_exp_text_matrix.py`) removes that bias by double-centering and
    # finds real discrimination on every embodiment (centered diagonal +1.40/+2.53/+1.64,
    # all CIs excluding 0; retrieval top-1 37.5% vs 6.2% chance). In the paired table the
    # bias cancels *exactly* -- pred and its own GT are scored against the same prompt in
    # 83/83 pairs -- and the paired delta is RESOLVED in the correct direction (GT better:
    # ctrlworld -0.552 [-1.019, -0.302], 4view -0.209 [-0.703, -0.066]). So it passes the
    # ceiling test that VisualIntegrity failed. Direction True = higher is more on-task.
    # Do NOT report unpaired/descriptive medians of this column across different captions.
    ('text_condition_alignment', 'Text Alignment (paired only)', True, 3),
    ('actor_motion_raw', 'Actor Activity', None, 2),
    ('actor_motion_active_rate', 'Active Rate', None, 3),
    ('active_motion_continuity', 'Active Motion Continuity', True, 2),
]
SUBJECT_KEYS = {'actor_motion_raw', 'actor_motion_active_rate',
                'active_motion_continuity', 'actor_flicker_raw'}
MIN_TRACKED_FRAC = 0.5
MIN_WEIGHT_SUM = 3.0
MIN_ACTIVE_PAIRS = 3
MIN_ACTIVE_FRAC = 0.05        # see run_dense_nr.MIN_ACTIVE_FRAC
# Head ego estimate needs this much background. This is a TABLE-TIME gate on purpose:
# E4 (2026-09-16) moved it 10 -> 5, and re-scoring 1,573 clips for a change that touches
# 29 records (1.84%) would be absurd -- and worse, it would split the results dir into
# two incomparable protocol versions right before the acceleration wave (E5) has to pair
# method clips against dense clips. Because `actor_per_frame` stores every per-pair
# input and `continuity_per_triple` stores every per-triple gain/copy error, all four
# subject columns are recomputable EXACTLY from the record (validated: max abs deviation
# 7e-15 on actor_motion_raw and 0.0 on active_rate over 1,520 clips when recomputed with
# the constant the run used). So the gate is applied here, uniformly, to old and new
# records alike -- which is what keeps v3.2 and v3.3 results comparable.
MIN_BG_COV_PCT = 5.0
ACTIVE_PX = 1.0
COPY_ERR_FLOOR = 2.0
MIN_VALID_PAIR_FRAC = 0.5
B = 10000

_SUBJECT_CACHE = {}


def subject_metrics(row):
    """Recompute the motion-derived columns from stored per-pair / per-triple records.

    Mirrors `run_dense_nr.actor_pair_motion` + `continuity`, but with THIS module's gate
    constants, so a results dir scored under any v3.x protocol aggregates under one
    consistent set of gates and a gate change needs no re-scoring.

    `active_rate` is a mean over the bg-coverage-gated motion dict, which on head mounts
    is strictly smaller than `actor_per_frame` -- so the activity floor is expressed
    against that dict, exactly as `run_dense_nr` does. (An earlier version reconstructed
    the count from `actor_motion_active_rate` because `actor_n_active_pairs` is absent on
    142 clips of the completed run; recomputing from the pair records removes the need
    and the two traps that came with it -- missing != 0, and the denominator is not
    `len(actor_per_frame)`.)
    """
    key = row.get('id')
    if key is not None and key in _SUBJECT_CACHE:
        return _SUBJECT_CACHE[key]
    pf = row.get('actor_per_frame') or []
    out = {'actor_motion_raw': None, 'actor_motion_active_rate': None,
           'active_motion_continuity': None, 'n_active': None}
    if pf:
        mount, iv = row['camera_mount'], row['source_interval']
        motion = {}
        for p in pf:
            if mount == 'static':
                motion[p['pair']] = p['fg_flow_p95']
            elif mount == 'wrist':
                motion[p['pair']] = p['frame_flow_med']
            elif p['bg_cov_pct'] >= MIN_BG_COV_PCT:
                motion[p['pair']] = p['fg_flow_ego_p95']
        motion = {k: v / iv for k, v in motion.items()}
        tracked = row.get('actor_tracked_pair_frac')
        if tracked is None:
            tracked = len(pf) / max(len(row.get('sampled_source_frames') or []) - 1, 1)
        if (row.get('actor_n_arm_dets') and len(motion) >= MIN_VALID_PAIR_FRAC * len(pf)
                and tracked >= MIN_VALID_PAIR_FRAC):
            vals = list(motion.values())
            n_active = sum(1 for v in vals if v >= ACTIVE_PX)
            out['actor_motion_raw'] = sum(vals) / len(vals)
            out['actor_motion_active_rate'] = n_active / len(vals)
            out['n_active'] = n_active
            tri = row.get('continuity_per_triple') or {}
            gains, copy_mse = tri.get('gain') or [], tri.get('copy_mse') or []
            num = denom = 0.0
            for j, g in enumerate(gains):
                v = [motion[k] for k in (2 * j, 2 * j + 1) if k in motion]
                w = min((sum(v) / len(v)) / ACTIVE_PX, 1.0) if v else 0.0
                if j < len(copy_mse) and copy_mse[j] < COPY_ERR_FLOOR:
                    w = 0.0
                num += w * g
                denom += w
            if (denom >= MIN_WEIGHT_SUM
                    and n_active >= max(MIN_ACTIVE_PAIRS, MIN_ACTIVE_FRAC * len(vals))):
                out['active_motion_continuity'] = num / denom * 100
    if key is not None:
        _SUBJECT_CACHE[key] = out
    return out


def gated(row, key):
    """Metric value under the current (v3.3) validity gates, or None.

    Recomputed here rather than trusted from the record so that clips scored by any
    v3.x protocol aggregate identically: v3.1 shipped neither activity gate, v3.2 used
    `MIN_BG_COV_PCT = 10`, v3.3 uses 5. For the three motion-derived columns the value
    itself is recomputed from the stored per-pair records (see `subject_metrics`), so
    changing a gate here does NOT require re-scoring.
    """
    value = row.get(key)
    if value is None and key not in SUBJECT_KEYS:
        return None
    if key in SUBJECT_KEYS:
        if row.get('actor_validity') != 'robot_detected':
            return None
        possible = max(len(row.get('sampled_source_frames') or []) - 1, 1)
        tracked = row.get('actor_tracked_pair_frac')
        if tracked is None:
            tracked = len(row.get('actor_per_frame') or []) / possible
        if tracked < MIN_TRACKED_FRAC:
            return None
        # `actor_flicker_raw` has no stored per-pixel inputs, so it cannot be
        # recomputed; it keeps the stored value under the tracking gate above.
        # The other three are recomputed under THIS module's constants.
        if key != 'actor_flicker_raw':
            recomputed = subject_metrics(row)
            if key not in recomputed:
                return None
            value = recomputed[key]
            if value is None:
                return None
            if key == 'active_motion_continuity' and (
                    row.get('active_motion_continuity_weight_sum') or 0.0) < MIN_WEIGHT_SUM:
                # Belt-and-braces: `subject_metrics` already applies MIN_WEIGHT_SUM to
                # its own recomputed weight sum; this also drops clips whose stored sum
                # was below the floor when `continuity_per_triple` is absent.
                return None
    return value


def median(values):
    v = sorted(values)
    n = len(v)
    return None if not n else (v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2]))


def boot_ci(values, seed=0):
    """Percentile bootstrap CI of the median. Reported, never hidden: at these n the
    CI half-width is 1-4 points, which is the same size as the effects."""
    if len(values) < 3:
        return None, None
    rng = random.Random(seed)
    meds = sorted(median(rng.choices(values, k=len(values))) for _ in range(B))
    return meds[int(0.025 * B)], meds[int(0.975 * B)]


def wilcoxon_p(diffs):
    """Two-sided Wilcoxon signed-rank via normal approximation (n >= 10).

    Paired and non-parametric: these metrics are bounded and skewed, so a t-test on
    the raw values is not appropriate.
    """
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n < 10:
        return None
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:                                    # average ranks within ties
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(r for r, x in zip(ranks, d) if x > 0)
    mean = n * (n + 1) / 4.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sd == 0:
        return None
    z = (w_plus - mean) / sd
    return math.erfc(abs(z) / math.sqrt(2.0))       # two-sided


def stratum_table(rows):
    """T1: descriptive per (model, embodiment, mount)."""
    out = []
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r['model'], r['embodiment'], r['camera_mount'])].append(r)
    for (model, emb, mount), g in sorted(groups.items()):
        rec = {'model': model, 'embodiment': emb, 'camera_mount': mount, 'n_clips': len(g)}
        for key, _, _, _ in COLUMNS:
            if key in PAIRED_ONLY:
                # Absolute medians of this column mix per-caption biases across clips
                # and would invite a between-model ranking that the data cannot support.
                continue
            vals = [v for v in (gated(r, key) for r in g) if v is not None]
            lo, hi = boot_ci(vals)
            rec[f'{key}_n'] = len(vals)
            rec[f'{key}_excluded_frac'] = round(1 - len(vals) / len(g), 3)
            rec[f'{key}_median'] = median(vals)
            rec[f'{key}_ci_lo'], rec[f'{key}_ci_hi'] = lo, hi
        out.append(rec)
    return out


def comparable_pair(a, b):
    """Reject a pair whose two clips are not the same measurement.

    A paired delta is only meaningful when the two clips differ ONLY by the method. This
    is not hypothetical: `sampling_dataset/dense/single_arm/output/multiview/dreamgen/`
    was generated with a different rollout config than the accel trees at the SAME
    relative path -- 277 frames @ 16 fps vs 93 frames @ 10 fps -- so 216/216
    dreamgen/single_arm accel pairs were silently comparing a 3x-longer 16fps dense clip
    against a 10fps accel clip. That produced a fake +10 AMC "improvement" for ALL NINE
    methods in that cell (radial, a near-no-op, showed AMC 72.4 vs dense 43.6), and it
    was the real cause of the fastercache AMC ceiling violation.

    AMC and the motion columns are per-source-frame quantities, so a different source fps
    or a 3x different clip length changes them regardless of method. `pair_key` cannot
    catch this: the two clips genuinely are the same episode+view, they were just
    RENDERED differently.

    Both fields are already stored by every record, so this gate is post-hoc and needs no
    rescoring. 226 of 1836 accel pairs (12.4%) are rejected, concentrated entirely in
    dreamgen/single_arm (100%) plus a handful in dreamdojo/single_arm and
    dreamgen/{humanoid,bimanual}.
    """
    if a.get('source_fps') != b.get('source_fps'):
        return False
    if a.get('effective_fps') != b.get('effective_fps'):
        return False
    na = len(a.get('sampled_source_frames') or [])
    nb = len(b.get('sampled_source_frames') or [])
    if not na or not nb:
        return False
    return abs(na - nb) / max(na, nb) <= MAX_LENGTH_MISMATCH


def resolved_flag(d, lo, hi, p, col, vs_dense):
    """Single definition of "this delta supports a claim", used by T2/T3/T3b.

    Three conditions, all measured-and-necessary:

    1. The bootstrap CI must STRICTLY exclude 0 (`lo > 0 or hi < 0`). The earlier
       `(lo > 0) == (hi > 0)` test passed a CI whose bound sat exactly ON 0 -- with
       discrete, tie-heavy columns like Active Rate that is common (e.g. svg1
       delta 0.000, CI [-0.059, 0.000]) and produced RESOLVED for a null result.
    2. n >= MIN_PAIRS_FOR_CLAIM, so the CI means something.
    3. The Wilcoxon p must not CONTRADICT the CI. The two disagree when the delta
       distribution is skewed: pisa Visual Integrity had median -0.68, CI
       [-1.25, -0.27] but p=0.262, i.e. the median shift is driven by part of the
       sample while the signed-rank sum is not significant. Claiming that is not
       defensible, so a large p vetoes. p is None below n=10 (normal approximation
       invalid); there the CI alone decides, as before.
    4. For a directional column in NO_IMPROVEMENT_VS_DENSE, a POSITIVE accel-vs-dense
       delta is not a claim -- the artifact that produces it is what the metric
       mis-rewards. Use artifact_positive() to surface those cells instead.
    """
    if not d or lo is None:
        return False
    if not (lo > 0 or hi < 0):
        return False
    if len(d) < MIN_PAIRS_FOR_CLAIM:
        return False
    if p is not None and p >= ALPHA:
        return False
    if col in CEILING_FAILED and not (vs_dense and col in CEILING_FAILURE_CANCELS_VS_DENSE):
        return False
    if vs_dense and col in NO_IMPROVEMENT_VS_DENSE and lo > 0:
        return False
    return True


def artifact_positive(d, lo, hi, p, col, vs_dense):
    """True when a cell is a statistically solid POSITIVE delta on a column where that
    direction cannot be an improvement (NO_IMPROVEMENT_VS_DENSE). resolved_flag()
    refuses to claim these; this reports them as what they are -- an over-smoothing
    flag on that (method, config) -- so they are never silently dropped."""
    if not (vs_dense and col in NO_IMPROVEMENT_VS_DENSE):
        return False
    if not d or lo is None or lo <= 0:
        return False
    if len(d) < MIN_PAIRS_FOR_CLAIM:
        return False
    if p is not None and p >= ALPHA:
        return False
    return True


def accel_deltas(rows, cell_of=None):
    """Paired (accel - dense) deltas per column, grouped by `cell_of(row)`.

    Shared by the per-cell (T3) and pooled-per-method (T3b) tables so both are
    guaranteed to be built from exactly the same paired deltas and the same gates --
    the pooled headline can never disagree with its own breakdown.
    """
    if cell_of is None:
        def cell_of(r):
            return (r['method'], r['model'], r['embodiment'], r['camera_mount'])
    dense, accel = {}, collections.defaultdict(list)
    for r in rows:
        if r.get('method') == 'dense':
            dense[(r['pair_key'], r['model'])] = r
        elif r.get('method') not in (None, 'gt'):
            accel[(r['pair_key'], r['model'])].append(r)
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    unpaired = collections.Counter()
    incomparable = collections.Counter()
    for key, variants in accel.items():
        base = dense.get(key)
        for r in variants:
            if base is None:
                unpaired[r['method']] += 1
                continue
            if not comparable_pair(r, base):
                incomparable[r['method']] += 1
                continue
            cell = cell_of(r)
            for col, _, _, _ in COLUMNS:
                a, b = gated(base, col), gated(r, col)
                if a is not None and b is not None:
                    groups[cell][col].append(b - a)
    return groups, unpaired, incomparable


def accel_pooled_table(rows):
    """T3b: acceleration method minus dense, POOLED over model/embodiment/mount.

    This is the headline accel table. Rationale (measured on the 9-method wave, 30
    paired clips each): splitting 30 clips over the real (model x embodiment x mount)
    grid yields n=1..7 per cell, so every cell is `ns` no matter how large the true
    effect is -- the per-cell grid has no power at any affordable budget. Pooling the
    same paired deltas per method restores n~20-30 and makes a claim possible.

    What pooling costs, and why it is acceptable here: each delta is still WITHIN one
    episode+view+geometry (accel vs its own dense clip), so the pooled median is a
    median of like-for-like differences, not a mix of absolute levels across
    embodiments. That is exactly the reason the NOT_CROSS_EMBODIMENT restriction does
    not bite here -- it forbids comparing absolute AMC LEVELS across embodiments, and
    a within-clip delta contains no level.

    Still report the per-cell T3 alongside: pooling hides a method that helps one
    (model, mount) and hurts another.
    """
    groups, unpaired, incomparable = accel_deltas(rows, cell_of=lambda r: r['method'])
    counts = collections.defaultdict(collections.Counter)
    for r in rows:
        m = r.get('method')
        if m not in (None, 'gt', 'dense'):
            counts[m][(r['model'], r['embodiment'], r['camera_mount'])] += 1
    out = []
    for method, per_key in sorted(groups.items()):
        rec = {'accel_method': method, 'n_cells_pooled': len(counts[method]),
               'n_clips': sum(counts[method].values())}
        for col, _, _, _ in COLUMNS:
            d = per_key.get(col, [])
            lo, hi = boot_ci(d)
            rec[f'{col}_n_pairs'] = len(d)
            rec[f'{col}_delta_median'] = median(d)
            rec[f'{col}_ci_lo'], rec[f'{col}_ci_hi'] = lo, hi
            rec[f'{col}_p'] = wilcoxon_p(d)
            rec[f'{col}_resolved'] = resolved_flag(
                d, lo, hi, rec[f'{col}_p'], col, vs_dense=True)
        out.append(rec)
    return out, unpaired, incomparable


def accel_table(rows):
    """T3: acceleration method minus the DENSE prediction of the same episode+view.

    This is the PhyFid-DR reference rule: an acceleration method is judged against
    dense, not against GT. It is also the only comparison with power -- a method's
    effect is smaller than the GT-pred gap, and unpaired medians already fail to
    resolve that larger gap.

    Pairing is on the method-free `pair_key`, so a row is only used when the dense
    record for the identical episode+view+geometry is present in the same results dir
    (or a dir aggregated with it). Cells are (method, model, embodiment, mount): a
    method's behaviour is model-specific, so pooling models would hide it.

    At a realistic accel budget (~30 clips/method) this per-cell grid fragments into
    n=1..7 cells that can never be RESOLVED -- use `accel_pooled_table` for the
    headline number and keep this table as the per-cell breakdown.
    """
    groups, unpaired, incomparable = accel_deltas(rows)
    out = []
    for cell, per_key in sorted(groups.items()):
        rec = {'accel_method': cell[0], 'model': cell[1],
               'embodiment': cell[2], 'camera_mount': cell[3]}
        for col, _, higher, _ in COLUMNS:
            d = per_key.get(col, [])
            lo, hi = boot_ci(d)
            rec[f'{col}_n_pairs'] = len(d)
            rec[f'{col}_delta_median'] = median(d)
            rec[f'{col}_ci_lo'], rec[f'{col}_ci_hi'] = lo, hi
            rec[f'{col}_p'] = wilcoxon_p(d)
            # "resolved" means the CI excludes 0, i.e. the method IS distinguishable
            # from dense on this column. A CI including 0 is a valid, expected result
            # ("not distinguishable at this n"), never a failure -- per the
            # acceleration-benchmark rule, do not drop or swap a method for it.
            rec[f'{col}_resolved'] = resolved_flag(
                d, lo, hi, rec[f'{col}_p'], col, vs_dense=True)
            # The head-mount motion confound applies to an accel delta exactly as it
            # applies to a GT delta: if an accel method changes how much the subject
            # appears to move, a confounded column moves with it for the wrong reason.
            warn = CONFOUNDED_MOUNTS.get(col, {}).get(cell[3])
            if warn:
                rec[f'{col}_confound_warning'] = warn
        out.append(rec)
    return out, unpaired, incomparable


def paired_table(rows):
    """T2: pred minus its own co-located GT, on identical episode+view+geometry."""
    by_key = collections.defaultdict(dict)
    for r in rows:
        # Keyed by (method, model), NOT model alone. `pair_key` is deliberately
        # method-free (that is what lets an accel clip find its dense partner), and an
        # accel row carries the SAME `model` as its dense counterpart -- e.g. a
        # `worldcache` clip of the ctrlworld tree has model 'ctrlworld'. Keying by model
        # alone therefore lets an accel row silently OVERWRITE the dense row for the
        # same episode+view and the GT delta would be computed on the wrong clip.
        by_key[r['pair_key']][(r.get('method'), r['model'])] = r
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for pair_key, entries in by_key.items():
        for (method, model), pred in entries.items():
            if method != 'dense':
                continue
            anchor = (entries.get(('gt', f'gt_{model}'))
                      or (entries.get(('gt', 'gt')) if model.startswith('ctrlworld') else None))
            if anchor is None:
                continue
            # Same requirement as the accel comparison: a pred-vs-GT delta is only
            # meaningful when both clips are the same measurement.
            if not comparable_pair(pred, anchor):
                continue
            cell = (model, pred['embodiment'], pred['camera_mount'])
            for key, _, _, _ in COLUMNS:
                a, b = gated(anchor, key), gated(pred, key)
                if a is not None and b is not None:
                    groups[cell][key].append(b - a)
    out = []
    for cell, per_key in sorted(groups.items()):
        rec = {'model': cell[0], 'embodiment': cell[1], 'camera_mount': cell[2]}
        for key, _, higher, _ in COLUMNS:
            d = per_key.get(key, [])
            lo, hi = boot_ci(d)
            rec[f'{key}_n_pairs'] = len(d)
            rec[f'{key}_delta_median'] = median(d)
            rec[f'{key}_ci_lo'], rec[f'{key}_ci_hi'] = lo, hi
            rec[f'{key}_p'] = wilcoxon_p(d)
            if higher is None:
                rec[f'{key}_gt_better_frac'] = None
            elif d:
                better = sum(1 for x in d if (x < 0 if higher else x > 0))
                rec[f'{key}_gt_better_frac'] = round(better / len(d), 3)
            else:
                rec[f'{key}_gt_better_frac'] = None
            # A claim is only permitted when the CI excludes 0, the cell has enough
            # pairs for that CI to mean anything, and the column did not fail the
            # GT-is-ceiling audit.
            rec[f'{key}_resolved'] = resolved_flag(
                d, lo, hi, rec[f'{key}_p'], key, vs_dense=False)
            mounts = CONFOUNDED_MOUNTS.get(key, {})
            rec[f'{key}_confound_warning'] = mounts.get(cell[2])
        out.append(rec)
    return out


def fmt(value, digits=2):
    return '--' if value is None else f'{value:.{digits}f}'


def write_csv(path, records):
    if not records:
        return
    # Union of keys, in first-seen order. Records are NOT homogeneous: a row only
    # carries `<key>_confound_warning` when that (column, mount) is flagged, so taking
    # fieldnames from records[0] raises ValueError as soon as a flagged row appears
    # later in the list.
    fields = list(dict.fromkeys(k for r in records for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)


def latex_paired(records, key, label, higher, digits):
    lines = [r'\begin{tabular}{lllrrl}', r'\toprule',
             r'Model & Embodiment & Mount & $n$ & $\Delta$ median [95\% CI] & Resolved \\',
             r'\midrule']
    for r in records:
        n = r[f'{key}_n_pairs']
        if not n:
            continue
        lines.append('%s & %s & %s & %d & %s [%s, %s] & %s \\\\' % (
            r['model'].replace('_', r'\_'), r['embodiment'].replace('_', r'\_'),
            r['camera_mount'], n, fmt(r[f'{key}_delta_median']),
            fmt(r[f'{key}_ci_lo']), fmt(r[f'{key}_ci_hi']),
            'yes' if r[f'{key}_resolved'] else 'no'))
    lines += [r'\bottomrule', r'\end{tabular}']
    direction = 'higher is better' if higher else ('lower is better' if higher is False else 'descriptive')
    header = f'% {label} ({direction}): pred minus own co-located GT\n'
    if key in CEILING_FAILED:
        header += '% WARNING: ' + CEILING_FAILED[key].replace('\n', '\n% ') + '\n'
    if key in NOT_CROSS_EMBODIMENT:
        header += '% WARNING (not cross-embodiment): ' + \
                  NOT_CROSS_EMBODIMENT[key].replace('\n', '\n% ') + '\n'
    return header + '\n'.join(lines)


def latex_accel_pooled(records, key, label, higher, digits):
    lines = [r'\begin{tabular}{lrrrl}', r'\toprule',
             r'Method & $n$ & cells & $\Delta$ median [95\% CI] & Resolved \\',
             r'\midrule']
    for r in records:
        n = r[f'{key}_n_pairs']
        if not n:
            continue
        lines.append('%s & %d & %d & %s [%s, %s] & %s \\\\' % (
            r['accel_method'].replace('_', r'\_'), n, r['n_cells_pooled'],
            fmt(r[f'{key}_delta_median'], digits), fmt(r[f'{key}_ci_lo'], digits),
            fmt(r[f'{key}_ci_hi'], digits),
            'yes' if r[f'{key}_resolved'] else 'no'))
    lines += [r'\bottomrule', r'\end{tabular}']
    direction = 'higher is better' if higher else ('lower is better' if higher is False else 'descriptive')
    header = (f'% {label} ({direction}): accel minus its own DENSE clip, pooled over\n'
              '% (model x embodiment x mount). Each delta is within one episode+view,\n'
              '% so no absolute level is compared across embodiments.\n')
    if key in ACCEL_CAVEATS:
        header += '% ACCEL CAVEAT: ' + ACCEL_CAVEATS[key].replace('\n', '\n% ') + '\n'
    if key in CEILING_FAILED and key not in CEILING_FAILURE_CANCELS_VS_DENSE:
        header += '% WARNING: ' + CEILING_FAILED[key].replace('\n', '\n% ') + '\n'
    return header + '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--also', type=Path, nargs='*', default=[],
                    help='additional run dirs to aggregate. Use this to pair an '
                         'acceleration run against the dense run that holds its '
                         'reference records, e.g. --run <accel_run> --also <dense_run>.')
    args = ap.parse_args()
    out = args.out or args.run / 'tables'
    out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(p.read_text()) for p in sorted((args.run / 'results').glob('*.json'))]
    if not rows:
        raise SystemExit('no results yet')
    manifest = json.loads((args.run / 'manifest.json').read_text())
    for extra in args.also:
        more = [json.loads(p.read_text()) for p in sorted((extra / 'results').glob('*.json'))]
        seen = {r['id'] for r in rows}
        rows += [r for r in more if r['id'] not in seen]
        manifest += json.loads((extra / 'manifest.json').read_text())
        print(f'aggregated {len(more)} records from {extra}')
    protocols = sorted({r.get('protocol_version') for r in rows})

    t1 = stratum_table(rows)
    t2 = paired_table(rows)
    t3, t3_unpaired, t3_incomp = accel_table(rows)
    t3b, _, _ = accel_pooled_table(rows)
    write_csv(out / 'table1_descriptive.csv', t1)
    write_csv(out / 'table2_paired_gt_delta.csv', t2)
    if t3:
        write_csv(out / 'table3_accel_vs_dense.csv', t3)
    if t3b:
        write_csv(out / 'table3b_accel_pooled.csv', t3b)
        for key, label, higher, digits in COLUMNS:
            (out / f'table3b_{key}.tex').write_text(
                latex_accel_pooled(t3b, key, label, higher, digits) + '\n')
    for key, label, higher, digits in COLUMNS:
        (out / f'table2_{key}.tex').write_text(latex_paired(t2, key, label, higher, digits) + '\n')
    # A previously-emitted table for a now-dropped column must not linger and get
    # picked up by the paper build.
    for key in DROPPED:
        (out / f'table2_{key}.tex').unlink(missing_ok=True)
        (out / f'table3b_{key}.tex').unlink(missing_ok=True)

    (out / 'provenance.json').write_text(json.dumps({
        'run': str(args.run), 'protocol_versions': protocols,
        'manifest_clips': len(manifest), 'scored_clips': len(rows),
        'errors': len(list((args.run / 'errors').glob('*.json'))),
        'gates_applied': {'min_tracked_pair_frac': MIN_TRACKED_FRAC,
                          'min_weight_sum': MIN_WEIGHT_SUM,
                          'min_active_pairs': MIN_ACTIVE_PAIRS,
                          'min_active_frac': MIN_ACTIVE_FRAC,
                          'min_bg_cov_pct': MIN_BG_COV_PCT,
                          'note': 'motion columns recomputed from stored per-pair '
                                  'records under these constants, so records scored '
                                  'under older gate values aggregate identically'},
        'protocol_version_counts': dict(collections.Counter(
            r.get('protocol_version') for r in rows)),
        'ceiling_failed_descriptive_only': CEILING_FAILED,
        'dropped_from_paper': DROPPED,
        'paired_only_columns': PAIRED_ONLY,
        'acceleration_caveats': ACCEL_CAVEATS,
        'min_pairs_for_claim': MIN_PAIRS_FOR_CLAIM,
        'confounded_mounts': CONFOUNDED_MOUNTS,
        'not_cross_embodiment': NOT_CROSS_EMBODIMENT,
        'pair_comparability_gate': {
            'max_length_mismatch': MAX_LENGTH_MISMATCH,
            'requires': ['source_fps', 'effective_fps', 'sampled_source_frames'],
            'note': 'A paired delta is dropped unless both clips share source_fps and '
                    'effective_fps and their sampled length differs by <=10%. Found '
                    '226/1836 accel pairs violating this, 216 of them the whole '
                    'dreamgen/single_arm cell, where the dense tree was rendered at '
                    '277 frames @ 16fps against accel at 93 frames @ 10fps. That was '
                    'producing a fake +10 AMC gain for all 9 methods in that cell.'},
        'accel_pooling_policy': (
            'Headline accel numbers (table3b) pool paired accel-minus-dense deltas '
            'over (model x embodiment x mount). Justified because each delta is '
            'within one episode+view+geometry, so no absolute level is compared '
            'across embodiments; the per-cell breakdown (table3) is reported '
            'alongside and is expected to be ns at ~30 clips/method.'),
        'bootstrap_resamples': B,
    }, indent=2))

    print(f'scored {len(rows)}/{len(manifest)} clips; protocols {protocols}')
    # Mixed versions are now only a *reporting* concern, not a correctness one: the
    # three motion columns are recomputed here from the stored per-pair records under
    # one set of constants, so v3.1/v3.2 records aggregate identically. Still surface
    # it, because a mix usually means two different sampling grids or probe versions,
    # which are NOT reconstructible.
    if len([p for p in protocols if p]) > 1:
        counts = collections.Counter(r.get('protocol_version') for r in rows)
        print('\n!! MIXED PROTOCOL VERSIONS in one results dir: '
              + ', '.join(f'{k}={v}' for k, v in sorted(counts.items(), key=lambda x: -x[1])))
        print('!! Gates are re-applied uniformly here, so the motion columns are still '
              'comparable; but verify the versions share the same sampling grid and '
              'probe before quoting anything.')
    for key, why in DROPPED.items():
        print(f'\n[DROPPED from paper] {key}: {why}')
    for key, why in PAIRED_ONLY.items():
        print(f'\n[PAIRED-ONLY] {key}: {why}')
    # Only relevant once an acceleration tree is in the run.
    accel = sorted({r.get('method') for r in rows} - {'dense', 'gt', None})
    if accel:
        print(f'\n[ACCELERATION METHODS PRESENT] {accel}')
        for key, why in ACCEL_CAVEATS.items():
            print(f'[ACCEL CAVEAT] {key}: {why}')
    print('\n== T2 paired (pred - own GT); only "resolved" rows support a claim ==')
    for key, label, higher, digits in COLUMNS:
        cells = [r for r in t2 if r[f'{key}_n_pairs']]
        if not cells:
            continue
        print(f'\n{label}:')
        if key in CEILING_FAILED:
            print('  !! DESCRIPTIVE ONLY -- ' + CEILING_FAILED[key])
        if key in NOT_CROSS_EMBODIMENT:
            print('  !! NOT CROSS-EMBODIMENT -- ' + NOT_CROSS_EMBODIMENT[key])
        for r in cells:
            p = r[f'{key}_p']
            warn = r.get(f'{key}_confound_warning')
            print('  %-22s %-10s %-6s n=%3d  delta %+9s [%+9s, %+9s]  %s%s%s' % (
                r['model'], r['embodiment'], r['camera_mount'], r[f'{key}_n_pairs'],
                fmt(r[f'{key}_delta_median'], digits), fmt(r[f'{key}_ci_lo'], digits),
                fmt(r[f'{key}_ci_hi'], digits),
                'RESOLVED' if r[f'{key}_resolved'] else 'ns',
                '' if p is None else f' p={p:.3g}',
                f'  !! CONFOUNDED ({warn})' if warn else ''))
    if t3b:
        print('\n== T3b HEADLINE accel - DENSE, POOLED per method '
              '(per-cell grid has no power at 30 clips/method) ==')
        if t3_unpaired:
            print('  (unpaired accel clips, no dense record at the same pair_key: '
                  + ', '.join(f'{k}={v}' for k, v in sorted(t3_unpaired.items())) + ')')
        if t3_incomp:
            print('  !! REJECTED as not-the-same-measurement (source_fps / effective_fps '
                  'differ, or length differs by >10%): '
                  + ', '.join(f'{k}={v}' for k, v in sorted(t3_incomp.items())))
            print('  !! The dense tree at that relative path was rendered with a '
                  'different rollout config than the accel tree, so comparing them '
                  'would measure the render, not the method. See comparable_pair().')
        for key, label, higher, digits in COLUMNS:
            cells = [r for r in t3b if r[f'{key}_n_pairs']]
            if not cells:
                continue
            print(f'\n{label}:')
            if key in ACCEL_CAVEATS:
                print('  !! ACCEL CAVEAT -- ' + ACCEL_CAVEATS[key])
            if key in CEILING_FAILED and key not in CEILING_FAILURE_CANCELS_VS_DENSE:
                print('  !! DESCRIPTIVE ONLY -- ' + CEILING_FAILED[key])
            for r in sorted(cells, key=lambda x: x[f'{key}_delta_median'] or 0):
                p = r[f'{key}_p']
                print('  %-12s n=%3d (%2d cells)  delta %+9s [%+9s, %+9s]  %s%s' % (
                    r['accel_method'], r[f'{key}_n_pairs'], r['n_cells_pooled'],
                    fmt(r[f'{key}_delta_median'], digits), fmt(r[f'{key}_ci_lo'], digits),
                    fmt(r[f'{key}_ci_hi'], digits),
                    'RESOLVED' if r[f'{key}_resolved'] else 'ns',
                    '' if p is None else f' p={p:.3g}'))
    if t3:
        print('\n== T3 accel - DENSE per (method x model x embodiment x mount) '
              'breakdown; expect ns at this budget -- see T3b for the claim ==')
        if t3_unpaired:
            print('  (unpaired accel clips, no dense record at the same pair_key: '
                  + ', '.join(f'{k}={v}' for k, v in sorted(t3_unpaired.items())) + ')')
        if t3_incomp:
            print('  !! REJECTED as not-the-same-measurement (source_fps / effective_fps '
                  'differ, or length differs by >10%): '
                  + ', '.join(f'{k}={v}' for k, v in sorted(t3_incomp.items())))
            print('  !! The dense tree at that relative path was rendered with a '
                  'different rollout config than the accel tree, so comparing them '
                  'would measure the render, not the method. See comparable_pair().')
        for key, label, higher, digits in COLUMNS:
            cells = [r for r in t3 if r[f'{key}_n_pairs']]
            if not cells:
                continue
            print(f'\n{label}:')
            if key in CEILING_FAILED:
                if key in CEILING_FAILURE_CANCELS_VS_DENSE:
                    print('  (GT-ceiling failure does NOT apply here: the '
                          'generative-vs-real bias cancels against the clip\'s own '
                          'dense counterpart -- E7 verdict 4. Claims allowed.)')
                else:
                    print('  !! DESCRIPTIVE ONLY -- ' + CEILING_FAILED[key])
            if key in ACCEL_CAVEATS:
                print('  !! ACCEL CAVEAT -- ' + ACCEL_CAVEATS[key])
            for r in cells:
                p = r[f'{key}_p']
                print('  %-12s %-22s %-10s %-6s n=%3d  delta %+9s [%+9s, %+9s]  %s%s' % (
                    r['accel_method'], r['model'], r['embodiment'], r['camera_mount'],
                    r[f'{key}_n_pairs'], fmt(r[f'{key}_delta_median'], digits),
                    fmt(r[f'{key}_ci_lo'], digits), fmt(r[f'{key}_ci_hi'], digits),
                    'RESOLVED' if r[f'{key}_resolved'] else 'ns',
                    '' if p is None else f' p={p:.3g}'))
                if r.get(f'{key}_confound_warning'):
                    print(f'      !! CONFOUNDED ({r[f"{key}_confound_warning"]})')
    print(f'\nwrote {out}')

if __name__ == '__main__':
    main()
