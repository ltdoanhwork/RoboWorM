"""E2 follow-up: caption-bias-free version of the prompt-shuffle control.

WHY THIS EXISTS (do not delete in favour of `_exp_text_shuffle.py`)

The single-shuffle control in `_exp_text_shuffle.py` compares
`CLIPScore(clip, own_prompt)` against `CLIPScore(clip, other_task_prompt)` and found
~0 gap, which was read as "CLIPScore is saturated -> drop TextAlignment". That design
is confounded, and the confound is measurable in its own output
(80-clip run, /tmp/dnr_audit/e2_full/clips.json):

    predictions only  n=40  delta +1.96 [+0.16, +2.93]   RESOLVED positive
    humanoid          n=31  delta -3.18 [-5.05, -0.96]   RESOLVED *negative*

A resolved NEGATIVE gap is impossible under saturation: if CLIPScore ignored the text
it would return the same number for both arms, not reliably *prefer the wrong prompt*.
The cause is that CLIPScore has a strong per-caption bias (style/length), and the
humanoid prompt pool mixes two registers -- long imperative teleop instructions
("pick up the pomegranate with left hand and put it in the storage box, and ...", ~20)
and short descriptive captions ("The robot's right hand places a cucumber into a white
tray", ~31). A within-embodiment shuffle therefore compares CAPTION STYLES (~7 points)
on top of image-text agreement (~2 points), and the pooled median is style noise.

FIX: score an N x N clip-by-prompt matrix per embodiment (diagonal = matched arm) and
DOUBLE-CENTER it (subtract row means and column means). Column centering removes each
caption's own bias; row centering removes each clip's own bias; what survives on the
diagonal is image-text agreement alone. Also report retrieval rank of the true prompt,
which is bias-free by construction because every candidate is scored on the same clip.

Cost: N^2 CLIPScore evaluations per embodiment on N clips (N=16 -> 256 per embodiment),
CLIPScore only -- no GroundingDINO, no SAM2, no flow.

    bash scripts/_exp_text_matrix.sh --out /tmp/e2_matrix --n 16 --device cuda
"""
import argparse
import collections
import json
import random
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from run_dense_nr import prepare, task_text, text_conditioned  # noqa: E402
from compute_phyworld import write_frames  # noqa: E402

QUAL_STRIDE = 5          # matches run_dense_nr.worker's score_qual(..., 5)


def boot_ci(values, b=10000, seed=0):
    if len(values) < 3:
        return None, None
    rng = random.Random(seed)
    meds = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(b))
    return meds[int(0.025 * b)], meds[int(0.975 * b)]


class ClipScore:
    """Exactly the CLIPScore path of metrics.quality.quality_control_probe.IQA."""

    def __init__(self, device):
        import torch
        from PIL import Image
        from torchvision import transforms
        from torchmetrics.multimodal.clip_score import CLIPScore
        self.torch, self.Image = torch, Image
        self.device = device
        self.pre = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
        self.m = CLIPScore(model_name_or_path='openai/clip-vit-base-patch16').to(device)

    def scores(self, path, captions):
        """All captions for one frame, so the image is preprocessed once."""
        with self.torch.no_grad():
            t = self.pre(self.Image.open(path).convert('RGB')).unsqueeze(0).to(self.device)
            u8 = (t[0] * 255).to(self.torch.uint8)
            return [float(self.m(u8, c).detach().item()) for c in captions]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path,
                    default=ROOT / 'metrics/WorldScore/results/dense_nr_v3_div30')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--n', type=int, default=16, help='matrix side per embodiment')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed', type=int, default=20260915)
    ap.add_argument('--predictions-only', action='store_true',
                    help='exclude GT/anchor clips (the paper column is about predictions)')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    rows = json.loads((args.run / 'manifest.json').read_text())
    pool = [r for r in rows
            if text_conditioned(r['model']) and task_text(r.get('instruction'))]
    if args.predictions_only:
        pool = [r for r in pool if not r['model'].startswith('gt')]
    per_emb = collections.defaultdict(list)
    for r in pool:
        per_emb[r['embodiment']].append(r)

    clip = ClipScore(args.device)
    out = {}
    for emb in sorted(per_emb):
        cand = per_emb[emb]
        rng.shuffle(cand)
        seen, picked = set(), []
        for r in cand:                       # one clip per distinct prompt
            t = task_text(r['instruction'])
            if t in seen:
                continue
            seen.add(t)
            picked.append(r)
            if len(picked) >= args.n:
                break
        if len(picked) < 6:
            print(f'SKIP {emb}: only {len(picked)} distinct prompts')
            continue
        prompts = [task_text(r['instruction']) for r in picked]
        n = len(picked)
        m = []
        for i, row in enumerate(picked):
            with tempfile.TemporaryDirectory(prefix='e2m_') as td:
                tmp = Path(td)
                arr, _ = prepare(row, tmp / 'analysis.mp4')
                paths = write_frames(arr, tmp / 'frames', stride=QUAL_STRIDE)
                acc = [0.0] * n
                for p in paths:
                    for j, v in enumerate(clip.scores(p, prompts)):
                        acc[j] += v
                m.append([v / len(paths) for v in acc])
            print(f'  {emb} row {i + 1}/{n} ({len(paths)} frames)', flush=True)

        row_mean = [statistics.fmean(r) for r in m]
        col_mean = [statistics.fmean(m[i][j] for i in range(n)) for j in range(n)]
        grand = statistics.fmean(row_mean)
        cen = [[m[i][j] - row_mean[i] - col_mean[j] + grand for j in range(n)]
               for i in range(n)]
        diag_raw = [m[i][i] for i in range(n)]
        off_raw = [m[i][j] for i in range(n) for j in range(n) if i != j]
        diag_cen = [cen[i][i] for i in range(n)]
        off_cen = [cen[i][j] for i in range(n) for j in range(n) if i != j]
        # Retrieval on column-centered scores: within one clip, which caption wins.
        colc = [[m[i][j] - col_mean[j] for j in range(n)] for i in range(n)]
        ranks = [sorted(range(n), key=lambda j: -colc[i][j]).index(i) + 1 for i in range(n)]
        lo, hi = boot_ci(diag_cen)
        out[emb] = {
            'n': n,
            'diag_raw_median': statistics.median(diag_raw),
            'off_raw_median': statistics.median(off_raw),
            'raw_gap': statistics.median(diag_raw) - statistics.median(off_raw),
            'diag_centered_median': statistics.median(diag_cen),
            'diag_centered_ci95': [lo, hi],
            'off_centered_median': statistics.median(off_cen),
            'centered_gap': statistics.median(diag_cen) - statistics.median(off_cen),
            'retrieval_top1': sum(1 for r in ranks if r == 1) / n,
            'retrieval_top1_chance': 1.0 / n,
            'retrieval_top3': sum(1 for r in ranks if r <= 3) / n,
            'median_rank_true_prompt': statistics.median(ranks),
            'chance_median_rank': (n + 1) / 2,
            'ranks': ranks,
            'prompts': prompts,
            'clip_ids': [r['id'] for r in picked],
            'models': [r['model'] for r in picked],
            'matrix': m,
        }
        o = out[emb]
        print(f'\n[{emb}] n={n}')
        print(f'  raw:      diag {o["diag_raw_median"]:.2f} vs off {o["off_raw_median"]:.2f} '
              f'(gap {o["raw_gap"]:+.2f})')
        print(f'  CENTERED: diag {o["diag_centered_median"]:+.3f} '
              f'[{lo:+.3f}, {hi:+.3f}] vs off {o["off_centered_median"]:+.3f} '
              f'(gap {o["centered_gap"]:+.3f})  '
              f'{"RESOLVED" if lo is not None and lo > 0 else "ns"}')
        print(f'  retrieval top-1 {o["retrieval_top1"]:.1%} (chance {1 / n:.1%}), '
              f'top-3 {o["retrieval_top3"]:.1%}, median rank {o["median_rank_true_prompt"]:.1f} '
              f'(chance {(n + 1) / 2:.1f})\n', flush=True)
        (args.out / 'matrix.json').write_text(json.dumps(out, indent=2))

    # Pooled verdict across embodiments.
    if out:
        alld = [c for e in out.values()
                for c in [e['diag_centered_median']]]
        pooled_top1 = statistics.fmean(e['retrieval_top1'] for e in out.values())
        pooled_chance = statistics.fmean(e['retrieval_top1_chance'] for e in out.values())
        print('== POOLED ==')
        print(f'  centered diagonal medians per embodiment: '
              f'{[round(x, 3) for x in alld]}')
        print(f'  mean retrieval top-1 {pooled_top1:.1%} vs chance {pooled_chance:.1%}')
        verdict = ('a: CLIPScore DOES discriminate once caption bias is removed '
                   '-> the single-shuffle drop decision must be revisited'
                   if pooled_top1 > 2 * pooled_chance else
                   'b: still at chance after removing caption bias -> DROP confirmed')
        print(f'  verdict: {verdict}')
        out['_pooled'] = {'mean_retrieval_top1': pooled_top1,
                          'chance': pooled_chance, 'verdict': verdict}
        (args.out / 'matrix.json').write_text(json.dumps(out, indent=2))
    print(f'wrote {args.out / "matrix.json"}')


if __name__ == '__main__':
    main()
