# Dense NR — experiments still needed before the numbers go in the paper

Hand this file to an agent. Each experiment is self-contained: why it exists, what
to run, and **the pass criterion that decides the outcome**. Run them in the order
listed — E1 gates everything else.

Repo: `/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics`
Protocol: `dense-nr-v3.2-closed-world`. Background: [dense_nr_full_run.md](dense_nr_full_run.md).
Second-GPU capacity: [dense_nr_second_gpu.md](dense_nr_second_gpu.md).

## Ground rules (violating these invalidates the result)

1. **Never edit `scripts/run_dense_nr.py`, `metrics/dynamics/*`, or anything the
   workers import while the main run is live.** Workers load the code once; a
   mid-run edit silently mixes two protocols in one results dir. Wait for E1, or
   work on a **copy** of the run dir.
2. **Never write experiment output into `metrics/WorldScore/results/dense_nr_v3_div30`.**
   Use a scratch `--out` and delete it after, per the repo's sweep-hygiene rule.
3. **GT is the ceiling.** Any metric change is only an improvement if GT still beats
   pred on paired episode-views. If a "fix" makes pred beat GT, the fix is wrong.
4. Report **median + 95% bootstrap CI + n**, never a bare mean. Judge with a visual
   check on extreme clips, not aggregates alone — the AMC-78 artifact was invisible
   in every median.

---

## E1 — Completion audit (BLOCKING, no new compute) — **DONE 2026-09-16: FAILED on VisualIntegrity**

> **Result.** Run finished 1572/1573. GT-is-ceiling FAILS for VisualIntegrity
> (dreamdojo pred−GT **+0.90, p=5e-05**; 4view +0.52; ctrlworld −0.02 coin-flip).
> Motion / smoothness / sharpness explanations were all tested and rejected.
> Decision: VisualIntegrity is **descriptive-only**, never `RESOLVED`.
> AMC and ActorFlicker pass the direction check. Full numbers in
> [dense_nr_full_run.md](dense_nr_full_run.md). **Do not re-run.**
>
> **Second pass, same run, criteria 3-5 (which the above did not score):**
> `rho` criteria **also FAIL, and only on head mounts** — AMC −0.541, VI −0.553
> (n≈208), vs the <0.35 bar. The v3.1 "AMC is decorrelated from motion (−0.27)" note
> predates head data. On head views AMC/VI partly measure motion amount → report them
> per mount with head flagged, or exclude head; never pool head into one AMC number.
> Criterion 5 is now **FIXED**: the `active_rate<0.05 & AMC>70` class went 3 → **0**
> clips after adding a length-aware `MIN_ACTIVE_FRAC = 0.05` (see below). Verdict at
> this point: **1/7 → 2/7 criteria pass**; only AMC (off head) survives as a claim.
>
> **Gate bug behind criterion 5.** `MIN_ACTIVE_PAIRS = 3` is an absolute count, so it
> is nearly free on long clips: the 3 survivors had 79/177/245 valid pairs and passed
> with 3-7 active pairs at `active_rate` 0.02-0.04, reporting AMC 71/71/82. Added
> `MIN_ACTIVE_FRAC = 0.05` (`n_active >= max(3, 0.05*n_valid)`) in `run_dense_nr.py`
> **and** in `table_dense_nr.gated()`, so all 1,572 scored clips are fixed with no
> re-scoring. Calibrated: 0.05 drops exactly 15 clips, all ≤ 0.67 px/source-frame,
> zero clips ≥ 1 px; 0.15 would start deleting real motion (4 clips ≥ 1 px).
> Two traps when reconstructing it: `actor_n_active_pairs` is **missing on 142 clips**
> (missing ≠ 0, or moving clips at rate 0.95 look inactive), and the denominator is the
> bg-gated motion dict (`actor_motion_valid_pair_frac`), **not** `len(actor_per_frame)`.

**Operational gate (this bit was wrong and cost 12 h of ambiguity).** Do not treat the
existence of `queue_finished.json` as completion — a SIGTERM'd queue writes it too. On
09-15 it appeared at 07:37 with `worker_exit_codes: [-15,-15,-15,-15,-15]`; the real
completion at 20:11 wrote `[0,0,0,0,0]`. **Check the exit codes.**

The main run must finish first; three things cannot be judged from partial results.

**Why.** At 170/1573 only **1** GT/pred pair had *both* halves scored, because the
queue reaches the two members of a pair at different times. So the single most
important property — *is GT actually the ceiling?* — is currently untested.

**Run** (after `results/dense_nr_v3_div30/queue_finished.json` appears):

```bash
python3 assets/scripts/table_dense_nr.py --run metrics/WorldScore/results/dense_nr_v3_div30
```

**Pass criteria.**

| check | pass |
|---|---|
| GT better on paired VisualIntegrity | ≥ 65% of pairs, CI excludes 0 |
| GT better on paired ActiveMotionContinuity | ≥ 65% of pairs |
| `rho(actor_motion_raw, AMC)` per mount | \|rho\| < 0.35 |
| `rho(actor_motion_raw, visual_integrity)` | \|rho\| < 0.35 |
| clips with `active_rate < 0.05` and `AMC > 70` | **0** |
| error rate | < 2% (1 known: `Invalid FPS` on a corrupt GT source video) |

If GT loses on a metric, that metric measures motion amount or mask area — stop and
report; do not publish it.

---

## E2 — Does TextAlignment measure anything? — **RESOLVED 2026-09-16: KEPT (paired only); the earlier DROP was WRONG**

> **Final result (this supersedes the drop below).** The single-shuffle control was
> itself confounded. The deciding test — `scripts/_exp_text_matrix.py`, an N×N
> clip-by-prompt matrix per embodiment, **double-centered** to remove each caption's
> bias and each clip's bias — says CLIPScore **does** discriminate:
>
> | embodiment | raw diag vs off | **centered** diag [CI] | off | retrieval top-1 (chance 6.2%) |
> |---|---|---|---|---|
> | bimanual | 26.94 vs 24.42 | **+1.398 [+0.291, +3.023]** | −0.044 | 37.5% |
> | humanoid | 27.28 vs 24.97 | **+2.527 [+1.586, +3.658]** | −0.128 | 43.8% |
> | single_arm | 25.19 vs 23.47 | **+1.635 [+1.203, +2.012]** | −0.186 | 31.2% |
>
> All three RESOLVED, mean retrieval top-1 **37.5% = 6× chance**, median rank 2 of 16.
> Interpretation **(a)** holds: the metric is not saturated.
>
> **And the paper table was never exposed to the bias in the first place.** In the
> paired design, a prediction and its own GT are scored against the **same prompt** —
> verified **83/83 pairs**, zero mismatches — so the per-caption bias cancels *exactly*
> in the delta. The paired result behaves the way a valid metric should, and passes the
> ceiling test VisualIntegrity failed:
>
> | model | n | paired delta (pred − own GT) | |
> |---|---|---|---|
> | ctrlworld | 20 | **−0.552 [−1.019, −0.302]** | RESOLVED, GT better |
> | ctrlworld_4view_grid | 27 | **−0.209 [−0.703, −0.066]** | RESOLVED, GT better |
> | ALL | 83 | −0.301 [−0.444, −0.118] | GT better |
>
> **Action taken.** `text_condition_alignment` is **removed from `DROPPED` and restored
> to `COLUMNS`** in `assets/scripts/table_dense_nr.py`, but registered in a new `PAIRED_ONLY`
> dict: it is emitted in T2 (paired) and **excluded from T1 (descriptive)**, because an
> absolute median pools different captions and would invite exactly the between-model
> ranking the data cannot support. `table2_text_condition_alignment.tex` is emitted again.
>
> **Why the single-shuffle control gave the wrong answer** (keep this — it is the
> methodological lesson): CLIPScore's per-caption style/length bias is ≈7 points while
> image-text agreement is ≈2, so a within-embodiment shuffle mostly compares *caption
> styles*. The humanoid pool mixes long imperative teleop instructions (`'pick up the
> pomegranate with left hand and put it in the storage box, and …'` → ~20) with short
> descriptive captions (`"The robot's right hand places a cucumber into a white tray"`
> → ~31). That produced a *resolved negative* gap on humanoid (−3.18 [−5.05, −0.96])
> and a resolved *positive* one on predictions (+1.96 [+0.16, +2.93]) — impossible under
> saturation, since a text-blind metric cannot reliably prefer the *wrong* prompt. The
> pooled ~0 was style noise cancelling content signal, not a flat metric.
> **Lesson: never test a metric's sensitivity with a manipulation that also changes a
> nuisance variable the metric is known to be sensitive to. Either match the nuisance
> (same caption, paired) or remove it (double-centering).**
>
> Reproduce (≈70 min on 1 GPU, 3 embodiments × 16 clips); raw matrices, per-clip scores
> and retrieval ranks are archived at
> `tmp/20261005_refactor/misc/worldscore_results/_experiments/e2_text_matrix.json`:
> ```bash
> bash scripts/_exp_text_matrix.sh --out /tmp/e2_matrix --n 16 --device cuda
> ```

<details>
<summary>Superseded: the original single-shuffle DROP decision (kept for the record)</summary>

> **Result.** n=100, `scripts/_exp_text_shuffle.py`: matched 24.781 vs shuffled
> 24.766, paired delta **+0.088 [−1.043, +0.930]**. Both pre-registered drop
> conditions met (gap < 1.0 AND CI includes 0) → interpretation (b).
> TextAlignment is **removed from the paper** (enforced by the `DROPPED` dict in
> `assets/scripts/table_dense_nr.py`). Raw data still computed and stored.
> **Do not re-run.**
>
> ⚠️ This conclusion was **overturned** by the matrix control above — the numbers here
> are correct but the *design* was caption-confounded. The strata that exposed it
> (independent 80-clip replication): predictions only +1.96 [+0.16, +2.93] RESOLVED
> positive, bimanual +1.66 [+0.42, +3.38] RESOLVED positive, **humanoid −3.18
> [−5.05, −0.96] RESOLVED negative**. Per-clip |delta| median 2.93 with 45% of clips
> over 3 points, while the win rate is 52% — large but sign-less, i.e. a nuisance
> variable, not a flat metric.

</details>

**Why this is the highest-value experiment.** Measured on 154 clips, the
between-model spread of TextAlignment medians is **1.14** while the within-model sd
is **2.81**. Every model sits at 24.8–26.0:

| model | n | min | median | max |
|---|---|---|---|---|
| ctrlworld | 22 | 19.9 | 25.0 | 30.7 |
| ctrlworld_4view_grid | 17 | 19.3 | 24.9 | 27.2 |
| dreamgen | 21 | 20.5 | 26.0 | 35.3 |
| gt | 25 | 20.8 | 25.6 | 31.0 |

The prompts are correct (the v3.1 task-clause fix holds — e.g. `'close toolbox'`,
`'fold bath towel'`), so this is not the old truncation bug.

**Already ruled out: an input-corruption bug.** `IQA.frame_scores` does
`u8 = (t[0] * 255).to(torch.uint8)` before calling torchmetrics CLIPScore, which
would produce a garbage image if `self.pre` normalized. It does not —
`self.pre = Compose([Resize((512,512)), ToTensor()])`, so values are [0,1] and the
uint8 cast is correct (and matches WorldScore's own wrapper). Do not re-investigate
this; the CLIPScore input is sound, which is precisely why the narrow spread has to
be explained by the model/metric and not by a preprocessing artifact.

But a column where between-model differences are 0.4 sd cannot rank models.
**Two very different explanations, and they lead to opposite paper decisions:**

- (a) CLIPScore is working and the models genuinely are equally on-task → keep the
  column, report it as paired deltas only (pairing removes episode-level variance).
- (b) CLIPScore is saturated on robot imagery and returns ~25 for any text → the
  column is meaningless and must be dropped.

**The discriminating test:** score each clip against a *mismatched* instruction taken
from a different episode. If (a), the shuffled score drops clearly. If (b), it does not
move.

> ⚠️ **This test as stated is not sound** — see the resolution at the top of E2. A
> mismatched instruction changes the caption's *style and length* as well as its
> content, and CLIPScore's per-caption bias (≈7 points) is larger than the image-text
> agreement it is meant to probe (≈2). Use the double-centered matrix
> (`scripts/_exp_text_matrix.py`) or a same-caption paired comparison instead. The
> single-shuffle numbers below are retained only to document how the wrong conclusion
> was reached.

**Run.** Write `scripts/_exp_text_shuffle.py`. Score each clip twice: with its own
`text_condition_prompt` and with a prompt sampled from a different task in the same
embodiment. **Do not call `score_qual`** — it also runs GroundingDINO (`det.rate`) on
every frame, which is the expensive part and is irrelevant here. Call only
`iqa.frame_scores(path, prompt)` — its **third** return value is the raw CLIPScore
that feeds `task_lock` (`task_lock_score` in `metrics/phyworld/scoring.py` is only a
0–100 clamp, so comparing raw CLIPScore is equivalent) — and reuse `task_text` from
`run_dense_nr.py` so prompts are preprocessed identically. Frames can be re-extracted
with `prepare()`. Use ~80 clips spread over models. Always go through
`scripts/run_dense_nr.sh` for the venv and vendored `PYTHONPATH`, or import the same
way it does.

**Gotcha that would silently invalidate this experiment.** `score_qual` calls
`iqa.frame_scores(p, instruction or "a robot performing a task")` — when the
instruction is empty it substitutes a **generic caption** rather than skipping, so a
clip with no task text still yields a CLIPScore, which is only hidden afterwards by
`text_condition_applicable = False`. Exclude clips whose `task_text` is empty (all
DreamDojo) and assert the caption you pass is non-empty. Otherwise part of your
"matched" arm is really the generic-caption fallback, the matched-vs-shuffled gap gets
diluted toward zero, and that looks exactly like interpretation (b) — you would
conclude "drop the column" from an artifact of your own harness.

**Pass criterion.** Paired `matched − shuffled` median **> 2.0** with a 95% CI
excluding 0 → interpretation (a), keep the column. If the gap is < 1.0 or the CI
includes 0 → interpretation (b): **drop TextAlignment from the paper** and say so
explicitly in the schema. Report the number either way.

---

## E3 — ActorFlicker: fix the motion confound or drop the column — **RESOLVED: DROP. The per-pixel fix was BUILT, RUN, and it does not save the column**

> **Verdict 2026-09-16: DROP, now confirmed from two independent directions.**
> E7 killed it on artifact *sign* (the metric improves under every degradation).
> E3 has now also been executed as specified — the per-pixel renormalisation was
> implemented and 53 clips re-scored — and **no candidate statistic passes the
> pre-registered criteria**. The column stays out of the paper. What follows is the
> evidence, because "we tried the obvious fix and it failed" is a much stronger
> justification than "we suspected it was confounded".
>
> ### What was built
>
> `metrics/dynamics/actor_flicker.py::residual_variants` computes **9 candidate
> statistics in ONE probe pass** plus 4 per-pixel diagnostics (`copy`, `flow`,
> `grad`, `align_pred`). One pass is the whole point: optical flow + SAM2 masks are
> what cost GPU time, the candidates are numpy on arrays that already exist, so a
> single re-scoring run decides between all of them instead of one run per candidate
> (same principle that made the E4 gate a table-time decision). `aligned_residual`
> (v1) is untouched and `actor_flicker_raw` was verified to reproduce the stored
> v3.2 value with **relative deviation 0.0e+00 on all 53 clips** — the baseline did
> not move under the refactor.
>
> Candidates: `flownorm` (r/(1+m) per pixel), `gradnorm` (r/|∇I|), `unexplained`
> (Σr/Σ|I₁−I₂|, the AMC-style motion-cancelling ratio), `shiftmin` (best residual
> within ±1 px), and `sub_K` for K ∈ {0.05, 0.1, 0.2, 0.4} (r − K·|∇I|·m, subtractive
> so it cannot invert the correlation the way dividing by clip motion did).
>
> ### Result: criterion 1 fails for every candidate (53 clips, n=19/19/10 per mount)
>
> | candidate | rho static | rho head | rho wrist | C1 |
> |---|---|---|---|---|
> | raw (v1) | +0.786 | −0.067 | +0.575 | FAIL |
> | flownorm | +0.321 | +0.115 | +0.272 | FAIL |
> | gradnorm | +0.830 | −0.430 | +0.612 | FAIL |
> | **unexplained** | **−0.079** | **+0.139** | +0.405 | FAIL (wrist) |
> | shiftmin | +0.791 | −0.333 | +0.623 | FAIL |
> | sub_0.4 | +0.419 | +0.115 | +0.389 | FAIL |
>
> Criterion 2 passed for all of them (GT better in 73–81% of 26 pairs) — which is
> exactly why criterion 1 exists: C2 alone cannot tell a flicker metric from a
> motion metric, since GT also moves less erratically.
>
> ### New insight 1 — the confound is `copy`, not `flow`
>
> With per-pair diagnostics stored, the driver is measurable directly (5,254 pairs):
>
> | mount | rho(flow, r) | **rho(copy, r)** | rho(grad, r) | rho(grad·flow, r) |
> |---|---|---|---|---|
> | static | +0.787 | **+0.904** | +0.125 | +0.792 |
> | head | +0.589 | **+0.861** | +0.527 | +0.638 |
> | wrist | +0.692 | **+0.889** | +0.242 | +0.711 |
>
> The residual is best explained by **how much the image changed at all** (`copy`),
> not by displacement or by gradient. That is why `gradnorm` and `sub_K` barely help:
> they correct for *misalignment*, but the residual is dominated by scene change
> that flow cannot follow at all (occlusion, appearance change, codec noise). Only
> `unexplained`, which divides by `copy`, moves the needle — and it is the one that
> nearly passed. **If anyone revisits this, normalize by observed temporal change,
> not by flow magnitude.** The original E3 instruction ("normalize by local flow
> magnitude per pixel") was aimed at the wrong quantity; `flownorm` implements it
> literally and still fails.
>
> ### New insight 2 — the post-hoc "winner" was noise, and the criterion is under-powered
>
> A zero-GPU search over 7 more candidates built from the stored diagnostics
> (`scripts/_exp_e3_posthoc.py`) produced one full pass: `ratio_med` =
> median(r/max(copy, 0.01)) — rho −0.128/+0.079/+0.191, GT better 69.6%, spread
> 1.85×. Two checks killed it:
>
> 1. **It only passes at exactly that arbitrary floor.** Sweeping the 0.01:
>    floor 0.005 → rho −0.698, 0.01 → −0.128, 0.02 → +0.475. A statistic whose
>    confound flips sign under a ±2× change of a nuisance constant is fitting noise.
> 2. **Multiple comparisons.** Permuting motion labels within mount, the chance that
>    **at least one of the 16 screened candidates passes C1 under the null is 0.87**.
>
> And the criterion itself is weak at this n. `|rho| < 0.25` is an **equivalence**
> test, so small n makes it easy to pass:
>
> | n per mount | P(false FAIL \| rho_true=0) | P(false PASS \| rho_true=0.35) |
> |---|---|---|
> | 10 (head, selection run) | 0.45 | 0.38 |
> | 19 | 0.29 | 0.34 |
> | 60 | **0.05** | 0.22 |
>
> So the pre-registered "~60 clips spanning all three mounts" was **60 total, which
> is ~19/mount — under-powered by design**. It needs ≥60 *per mount*.
>
> ### Confirmation run (out-of-sample, 60/mount) — **DONE. It reverses the interim read twice, and the third test kills the survivor.**
>
> 305 clips, 0 errors, `--exclude-run` holding out the 54 selection clips.
>
> **At full power one candidate DOES pass the two pre-registered statistical criteria:**
>
> | candidate | rho head | rho static | rho wrist | C1 | C2 (GT better) | spread |
> |---|---|---|---|---|---|---|
> | raw (v1) | +0.405 | +0.728 | +0.361 | FAIL | 58.4% FAIL | 4.56× |
> | flownorm | −0.055 | +0.556 | +0.169 | FAIL | 61.6% FAIL | 2.86× |
> | **unexplained** | **−0.094** | **+0.232** | **+0.135** | **PASS** | **84.8% PASS** | 1.86× |
> | gradnorm | +0.796 | +0.750 | +0.399 | FAIL | 65.6% PASS | 4.18× |
> | sub_0.4 | +0.368 | +0.570 | +0.222 | FAIL | 68.0% PASS | 3.11× |
>
> Note how the numbers moved from the 53-clip selection run: `unexplained` static went
> **−0.079 → +0.232** and its C2 **76.9% → 84.8%**, while `raw`'s C2 **collapsed
> 80.8% → 58.4%**. At the interim 131-clip mark the same candidate read **+0.272 static /
> 55.6% C2 = FAIL**. Three different verdicts from three sample sizes of the same
> statistic — the single strongest argument in this whole document for not concluding at
> low n.
>
> **Three robustness tests, and it fails all three:**
>
> 1. **Arbitrary-constant sensitivity.** `unexplained` divides by `max(copy, 0.01)`.
>    Sweeping that floor: 0.005 → rho −0.386, **0.01 → −0.129**, 0.02 → +0.324,
>    0.05 → +0.706. It passes only in a narrow window around the value that happened to
>    be hardcoded — the identical failure mode that killed `ratio_med`.
> 2. **Leave-one-model-out.** Drop `ctrlworld` and static/wrist go **+0.310 / +0.323**;
>    drop `ctrlworld_4view_grid` and head goes **−0.303**. The same statistic passes or
>    fails depending on which model is in the sample, so the pass is a property of the
>    *mix*, not of the metric.
> 3. **Multiple comparisons.** Permuting motion labels within mount, the probability
>    that at least one of the 9 probe candidates passes C1 under the null is **1.00**
>    (0.87 on the selection set). At this n the C1 equivalence bar is passed by *chance*
>    almost surely — it has no discriminating power left in either direction.
>
> **And the decisive test — artifact sign (E7 protocol, 70 clips, re-run with the
> variants flattened in):**
>
> | artifact (mimics) | ActorFlicker (v1) | flick:flownorm | **flick:unexplained** |
> |---|---|---|---|
> | `dup2` frame reuse (**caching**) | −0.0038 ns | −0.0004 ns | **−0.0425 [−0.1347, −0.0025] ✗ REWARDS** |
> | `smooth3` temporal smoothing | ns | ns | +0.0984 ns |
> | `blur` σ1.6 (pruning) | −0.0049 ✗ | ns | −0.0494 ns |
> | `blur_mild` σ0.8 | −0.0024 ✗ | −0.0009 ✗ | ns |
> | `blur_real` σ0.45 (realistic) | −0.0010 ✗ | −0.0003 ✗ | ns |
> | `block` 8× (sparse attn) | ns | ns | +0.1237 ✅ |
>
> **`unexplained` "improves" when every second frame is duplicated** — precisely the
> artifact the step-caching methods (WorldCache/DiCache/FasterCache) produce, and the
> one AMC detects best (−39.3). A flicker metric that rewards frame reuse would credit
> the fastest cache configurations for being the most degraded. It is also blind to blur
> at all three severities. So the candidate that survived the statistics fails the
> mechanism check, and `flownorm` (the literal reading of the original E3 prescription)
> is significantly wrong on blur at realistic severity.
>
> **Final verdict: DROP, and the drop is now proven three ways** — confounded (v1),
> statistically unstable under any correction (all 16 candidates), and directionally
> wrong on the artifacts that matter (both survivors). This is a much stronger
> paper statement than "we suspected it was confounded".
>
> ### Reproduce
>
> ```bash
> python scripts/_exp_e3_build_manifest.py --pairs-per-mount 10   # selection sample
> bash scripts/run_dense_nr.sh --out /tmp/e3_flicker/run --workers 3 --reuse-manifest
> python scripts/_exp_e3_verdict.py       # C1 / C2 / power check
> python scripts/_exp_e3_posthoc.py       # zero-GPU search over stored diagnostics
> python scripts/test_actor_flicker.py    # unit tests incl. the flow-error confound
> ```
>
> The unit test now reproduces the confound synthetically and is worth keeping: with
> **exact** flow every candidate is already motion-independent (ratio 1.00), but with
> an 8% flow-estimation error and **no flicker at all**, `raw` inflates **5.02×** from
> 1 px to 6 px of displacement. That isolates the mechanism — the confound is flow
> *error*, not motion per se.
>
> ### The reusable lesson: three sample sizes gave three verdicts
>
> The same statistic (`unexplained`) read FAIL at n=19/mount, FAIL at n=44/mount, and
> PASS at n=60/mount; `raw`'s C2 went 80.8% → 58.4% over the same growth. Any
> criterion in this benchmark that is an **equivalence** bar (`|rho| < x`,
> "indistinguishable from dense") must have its n justified *before* it is read, and
> re-read at full n — the permutation check went 0.87 → 1.00, i.e. at large n the C1
> bar is passed by chance almost surely and carries no information either way. The
> three tests that actually separated signal from noise here were **arbitrary-constant
> sensitivity**, **leave-one-model-out**, and **artifact sign** — none of which is a
> significance test.
>
> ### Kept in the code (deliberately)
>
> `residual_variants` and the stored `actor_flicker_variants` stay. The cost is a few
> numpy ops, and they make this drop auditable and any future re-examination
> zero-GPU. `actor_flicker_raw` remains the reported (experimental, non-quotable)
> value so nothing downstream changes.

> **Original E7 finding (independent, and the stronger reason):** ActorFlicker goes
> DOWN ("improves") for all four synthetic degradations — frame duplication (−0.004),
> temporal smoothing, spatial blur (−0.005) and 8× blocking (−0.002). A metric that
> improves under every artifact cannot be renormalised into correctness, because the
> artifact itself suppresses the residual it measures. `scripts/_exp_accel_sanity.py`
> now also flattens `actor_flicker_variants`, so re-running it re-tests **every** new
> candidate against the artifact sign in one pass:
>
> ```bash
> python scripts/_exp_accel_sanity.py --build --n 8 --out /tmp/accel_sanity
> bash scripts/run_dense_nr.sh --out /tmp/accel_sanity --workers 2 --reuse-manifest
> python scripts/_exp_accel_sanity.py --report --out /tmp/accel_sanity
> ```
>
> **Earlier pre-check (kept — it is what motivated the per-pixel design):**
>
> | mount | rho raw | rho ÷motion | rho log-OLS residual, pair level |
> |---|---|---|---|
> | head | +0.473 | −0.784 | **−0.071** ✅ |
> | static | +0.695 | −0.794 | **+0.061** ✅ |
> | wrist | +0.364 | −0.778 | **−0.257** ✗ |
>
> The confound is almost purely **multiplicative** (log-log slope 0.19–0.27 per
> mount), which is why plain division overshoots. The strongest *post-hoc* correction
> on aggregates brings head and static inside the bar but wrist still fails, which is
> what made per-pixel work mandatory. That work is now done, and it also fails.

**Why.** ActorFlicker tracks motion amount, which is exactly what it must not do.
Measured, and **worse than the +0.45 recorded at v3.1**:

| mount | rho(motion, flicker) | rho(motion, flicker/motion) |
|---|---|---|
| static | **+0.75** | −0.81 |
| head | +0.45 | −0.89 |
| wrist | +0.51 | −0.85 |

The third column is the important part: **plain division overshoots**, flipping the
correlation from strongly positive to strongly negative. So do NOT "fix" it by
dividing by motion — that just inverts the bias. A fast-moving subject is not a
flickering one; the residual has to be normalized by the *local* flow magnitude at
each pixel, inside the probe, before it is averaged.

The column is already labelled `experimental ... until human-calibrated` in
[nr_paper_output_schema.md](nr_paper_output_schema.md), so nothing published so far
is wrong — but it **must not be quoted as a quality result** in its current state.

**Run.** In a *copy* of `metrics/dynamics/motion_accuracy_gdino.py`, implement the
flicker residual as a per-pixel ratio to local flow magnitude (or regress the
residual on flow magnitude and keep the residual-of-the-residual), then re-score
~60 clips spanning all three mounts to a scratch `--out`.

**Pass criteria (all three, or the column is dropped).**
1. `|rho(actor_motion_raw, flicker_v2)| < 0.25` on **every** mount.
2. GT still beats pred on paired flicker in ≥ 65% of pairs.
3. On 5 hand-picked clips with visible shimmer and 5 visually clean ones, the metric
   orders them correctly. Without this visual check the other two can be satisfied by
   a metric that measures nothing.

---

## E4 — `MIN_BG_COV_PCT` on head mounts — **DONE 2026-09-16: set to 5, ZERO GPU needed**

> **Result.** Answered fully **post-hoc** on all 241 head clips of the completed run —
> the "re-score ~40 clips at 3 settings" plan below is wasted GPU time, because
> `actor_per_frame` already stores per-pair `bg_cov_pct` and `fg_flow_ego_p95` and this
> gate is applied *after* the probe, so it can be re-applied exactly from the records.
>
> | `MIN_BG_COV_PCT` | usable head clips | None rate | median ActorActivity |
> |---|---|---|---|
> | **5** | **224** | 0.071 | 6.79 |
> | 10 (old) | 209 | 0.133 | 6.25 |
> | 20 | 183 | 0.241 | 6.62 |
>
> On a **fixed clip set** 10 → 5 moves the head median by **0.00%** (max 0.00%), far
> inside the <10% bar — the discarded pairs sit at `bg_cov < 5`, and the band [5,10) is
> only 5.66% of head pairs. So the gate does not bias values, it **discards whole
> clips**: 5 recovers **15 head clips (+6.2%)** for free. `thr=20` is where values do
> shift (per-clip max +34%). **Verdict: 5.**
>
> **How it was applied matters more than the value.** First attempt changed
> `MIN_BG_COV_PCT` in `run_dense_nr.py` and bumped `PROTOCOL` to v3.3 — which was wrong:
> the change touches **29 of 1,572 records (1.84%)**, so re-scoring 1,573 clips (~9 h)
> to realise it is absurd, and worse, it would split the results dir into two
> incomparable versions exactly when E5 needs to pair method clips against dense clips.
>
> Instead the gate was moved to **table time**. `actor_per_frame` stores every per-pair
> input (`bg_cov_pct`, `fg_flow_ego_p95`, `fg_flow_p95`, `frame_flow_med`) and
> `continuity_per_triple` stores every per-triple `gain`/`copy_mse`, so
> `table_dense_nr.subject_metrics()` recomputes `actor_motion_raw`,
> `actor_motion_active_rate` and `active_motion_continuity` from the record.
> **Validated exact** against the run's own constants over 1,520 clips:
>
> | column | compared | max abs deviation |
> |---|---|---|
> | `actor_motion_raw` | 1520 | 7.1e-15 |
> | `actor_motion_active_rate` | 1520 | 0.0 |
> | `active_motion_continuity` | 1451 | 9.9e-14 |
> | `..._weight_sum` | 1520 | 8.5e-14 |
>
> The one disagreement is the recompute being *more* correct: clip `17dcc9e8…` stores
> AMC 36.3 with **0** active pairs (the artifact class). Only 1 of 1,572 records lacks
> `actor_per_frame`, and it has no stored value either → no data is lost. Consequence:
> `PROTOCOL` stays at **v3.2**, both new gates apply to all already-scored clips, and
> either value is now a one-line switch (`MIN_BG_COV_PCT` in `table_dense_nr.py`).
> `actor_flicker_raw` is the exception — no per-pixel inputs are stored, so it cannot be
> recomputed (which is also why E3 needs GPU).
>
> **Limitation no compute can fix:** the paired-GT-delta half of the criterion has at
> most **n=12 head pairs (thr 5) / n=9 (thr 10)** in the entire manifest
> (+0.014 [−0.857, +0.623] vs −0.694 [−2.442, +1.810]) — state it as a limitation
> rather than waiting for data that does not exist.

**Why.** v3.1 recorded this gate as "currently inert" because no head-mount clip had
been scored. Head data now exists and the gate **does** bind:

| mount | pairs | bg_cov_pct p1 | p10 | median | % below 10 |
|---|---|---|---|---|---|
| static | 8223 | 1.3 | 33.7 | 70.3 | 4.4% |
| **head** | **1705** | **0.5** | **8.9** | 34.7 | **12.9%** |
| wrist | 5816 | 1.5 | 11.8 | 47.9 | 8.6% |

12.9% of head pairs are discarded. Head is also the mount with the largest motion
(median 6.2 px vs 2.9 static, 1.8 wrist), so silently dropping its low-background
pairs biases head ActorActivity — and only head uses ego-translation removal.

**Run.** Re-score ~40 head-mount clips at `MIN_BG_COV_PCT` ∈ {5, 10, 20} to scratch
dirs. Report per setting: `actor_motion_raw` median, `actor_tracked_pair_frac`, the
`None` rate, and the paired GT-delta.

**Pass criterion.** Keep 10 only if moving to 5 changes the head `actor_motion_raw`
median by < 10% **and** does not degrade the paired GT-delta. If 5 is equally good it
is strictly preferable (12.9% more data). If the three settings disagree materially,
head ActorActivity is gate-sensitive and that must be stated as a limitation.

---

## E5 — Acceleration wave (the actual paper table), 30 clips/group — **READY TO RUN: the code is written, the manifest verified, and a real pairing BUG was found and fixed**

> **Status 2026-09-16.** The two prerequisites are done and one blocker was found
> *because* they were done. Nothing is left but GPU time.
>
> ### 1. Sampler — done, and no new code in `inventory()` was needed
>
> `inventory(out, methods=...)` and `_pred_rows(tree, method, ...)` already
> parameterise by method tree, and `row_for` already namespaces the record `id` by
> method while keeping `pair_key` **method-free** (which is what pairs an accel clip
> to its dense counterpart). What was missing is the *sampler*, now
> `scripts/_exp_e5_build_manifest.py`: it restricts candidates to clips whose dense
> counterpart **already has a scored result**, samples 30 per
> (method, embodiment, mount) with `spread_sample` over (task, split), and emits
> **only** the method rows — GT is not re-emitted because the accel trees ship
> byte-identical copies and the dense partners are already scored.
>
> Verified output:
>
> ```
> dense reference: 1572 scored records, 733 pairable pair_keys
>   worldcache  pairable 864 -> 210 over 7 groups     itm   615 -> 210
>   dicache     pairable 864 -> 210                   sito  615 -> 210
>   fastercache pairable 358 -> 210                   pisa  676 -> 210
>   radial      pairable 674 -> 210                   svg1  702 -> 210
>   svg2        pairable 673 -> 210
> groups: 63, short of 30: 0
> TOTAL 1890 clips ~= 49 GPU-h (~9.9 h wall on 5 workers)
> ```
>
> **Correction to an earlier number in this doc: it is 49 GPU-h, not 12.6.** 1,890 ×
> 94 s = 49.4 GPU-h; the 9.9 h figure is *wall* time on 5 workers. Also note it is
> **7 groups per method, not 9** — only 7 (embodiment, mount) combinations exist in
> the dense NR sample: bimanual static/wrist, single_arm static/wrist, humanoid
> head/static/wrist.
>
> ### 2. BUG FOUND — the paired table would have silently used the wrong clip
>
> `paired_table` keyed its pairing dict by `model` alone:
>
> ```python
> by_key[r['pair_key']][r['model']] = r      # BUG
> ```
>
> `pair_key` is deliberately method-free, and an accel row carries the **same
> `model`** as its dense counterpart (a `worldcache` clip of the ctrlworld tree has
> `model == 'ctrlworld'`). So the moment an accel run is aggregated, the accel row
> **overwrites the dense row** for the same episode+view and the GT delta is computed
> on the wrong clip — silently, with no error and a plausible-looking number. Fixed
> to key by `(method, model)`. Verified this is latent-only today: the dense-only run
> has **0 collisions** on `(pair_key, method, model)` and 0 cases of one
> `(pair_key, model)` carrying two methods — so no published dense number is
> affected. (11 `pair_key`s *do* carry two different dense models, `ctrlworld` and
> `ctrlworld_4view_grid` — harmless, since those differ in `model`.)
>
> ### 3. T3 accel-vs-dense table — implemented
>
> `table_dense_nr.py` gained `accel_table()` (T3): pairs each method clip against the
> **dense prediction** at the same `pair_key` (PhyFid-DR reference rule, *not* GT),
> cells = (method, model, embodiment, mount), with bootstrap CI, Wilcoxon p and a
> `resolved` flag. It prints `ACCEL_CAVEATS` inline — notably that **AMC is
> blur-blind**, so a small positive AMC delta for a blur-like method (sito 0.758×,
> itm 0.741× sharpness vs dense) must not be read as "better than dense". Unpaired
> accel clips are counted and reported rather than dropped silently. New `--also`
> flag aggregates the dense reference run:
>
> ```bash
> python scripts/_exp_e5_build_manifest.py            # writes results/accel_nr_v1/manifest.json
> bash scripts/run_dense_nr.sh --out metrics/WorldScore/results/accel_nr_v1 \
>      --workers 5 --reuse-manifest
> python assets/scripts/table_dense_nr.py --run metrics/WorldScore/results/accel_nr_v1 \
>      --also metrics/WorldScore/results/dense_nr_v3_div30
> ```
>
> ### 4. Which columns E5 may report
>
> Only 3 of the 6 survive the audits, and this constrains E5 before it runs:
> **VisualIntegrity** and **AMC** (with the blur caveat and the head-mount motion
> confound), **ActorActivity** as descriptive, **TextAlignment** paired-only.
> **ActorFlicker must not appear in T3** (E3/E7: it *improves* under every synthetic
> artifact, so it would credit the very methods it should penalise).

> **Prerequisite VERIFIED 2026-09-16 — the pairing exists and the budget is 9.9 h
> wall, not 1.6 days.** The binding constraint is not "does the method mp4 exist" but
> "does the same relative path already have a **dense NR result**", and dense NR
> results exist only for the 1,573-clip div30 sample (not all 10,465 dense videos).
>
> - Method mp4s whose relative path also exists under `dense/`: **100%** for
>   worldcache, dicache, itm, sito, svg2; 99.6–99.9% pisa/radial; 97.1% svg1;
>   89.3% fastercache.
> - Against clips that **already have a dense NR result**: **0 of 63
>   (method × embodiment × mount) groups is short of 30** pairable clips.
> - Realistic budget **1,890 clips ≈ 49 GPU-h (≈9.9 h wall on 5 workers
>   at the measured 94 s/clip median)**.
>
> So E5 needs no extra dense scoring — only the code change in the Prerequisite below.
> Safe to make now: the div30 queue has exited (`worker_exit_codes: [0,0,0,0,0]`).
> The E4 gate change costs E5 **nothing**: it is applied at table time from the stored
> per-pair records (see E4), so method clips scored now pair directly against the
> existing 1,573 dense results with no re-scoring and no protocol split.

**Why.** Full NR scoring of the 9 acceleration trees is impossible: 80,174
prediction videos ≈ 173,913 view-clips ≈ **48 days** on one GPU at the measured
24 s/clip. Agreed scope is **30 clips per (method, embodiment, camera_mount)** ≈
5,670 clips ≈ **1.6 days**.

**Design constraint that must not be relaxed.** The reference for an acceleration
method is the **`dense` prediction of the same episode+view**, not GT — this is the
PhyFid-DR rule, and it is also the only thing with statistical power: an
acceleration method's effect is *smaller* than the GT−pred gap, and unpaired medians
already fail to resolve that larger gap (VisualIntegrity unpaired CIs overlap:
GT 32.53 [31.04, 33.60] vs pred 31.66 [29.71, 33.11], while paired resolves at
+1.00 [+0.56, +1.60]).

**Prerequisite.** ~~`dense` must be scored for exactly the episode+views sampled for
each method~~ — **DONE**, see the status block above: `inventory()`/`_pred_rows()`
already take a method tree, and `scripts/_exp_e5_build_manifest.py` restricts the
sample to episode+views that already have a dense result.

**Pass criterion.** Every method row in **T3** (`table3_accel_vs_dense.csv`) reports n,
a paired delta vs **dense**, a CI, and a `resolved` flag. Methods whose CI includes 0
are reported as "not distinguishable from dense at this n" — which is a valid and
expected result, not a failure. Per the acceleration-benchmark rule, do **not** drop
or swap a method that scores badly.

---

## E6 — Cross-fps comparability of ActorActivity (lower priority)

**Why.** `actor_motion_raw` is px/source-frame on a fixed 5 fps analysis grid, so
`source_interval` differs per model (GT and both Ctrl-World 1, DreamDojo 2 /
single_arm 3, DreamGen 3). The v3.2 per-model co-located anchor removed this problem
**for the paired column** (anchor and pred share fps, resolution and frame count
exactly), but T1 descriptive still puts different-fps models in adjacent rows.

**Run.** Take ~30 GT clips, score each at analysis fps ∈ {5, 10} and compare
`actor_motion_raw` vs `actor_motion_px_per_second`.

**Pass criterion.** If `actor_motion_px_per_second` is stable across analysis fps
(median change < 10%) while `actor_motion_raw` is not, then T1 must report
px/second, and `actor_motion_raw` stays an internal field only.

---

## Not an agent task

Threshold calibration (`ACTIVE_PX`, the flicker threshold, predicate detectors)
freezes from **held-out human labels**, per the schema. Until those exist, publish
native-unit distributions — never an invented 0–100 quality map. `ACTIVE_PX = 1.0`
currently looks reasonable and does **not** need an experiment: no mount saturates
(median active rate 0.62 static / 0.88 head / 0.42 wrist; rate=1 in only 0–10% of
clips), so weights are neither all-ones nor all-zeros. Per-triple VFI gains are also
healthy (p10 0.21, median 0.61, p90 0.89; 0.0% at the upper clip, 2.4% at the lower),
so the `clip(0,1)` is not destroying resolution.

## Where to write results

Append a dated section to [dense_nr_full_run.md](dense_nr_full_run.md) with the
measured numbers and the pass/fail verdict, and update the frozen column list in
[nr_paper_output_schema.md](nr_paper_output_schema.md) if a column is dropped or
redefined. Record negative results — a column that had to be dropped is a finding.

<!-- E3 raw numbers (selection 53, confirmation 305, artifact-sign 70) are archived at
     tmp/20261005_refactor/misc/worldscore_results/_experiments/e3_flicker_variants.json (0.35 MB) so the
     drop stays auditable after the scratch run dirs were deleted. -->
