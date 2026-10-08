# Third-party code and licenses

RoboWorM vendors, forks or links several upstream projects. This page lists every one
of them: where it lives in the repo, where it came from, its license, and what we
changed. Third-party code keeps its upstream license. The location of each license text
is given in the **license file** column.

How code enters the repo:

- **vendored**: upstream files copied into the tree and tracked here.
- **gitlink**: a nested git checkout recorded as a commit pointer. This repo has no
  `.gitmodules` entries for them yet, so a plain `git clone` gives an **empty directory**.
  Clone the upstream URL at the listed commit into that path (commands at the end of
  this page).
- **pip**: a package installed into one of the venvs ([setup/README.md](setup/README.md)).
  It is not stored in the repo, and it is listed here because a paper metric depends on it.
- **re-implemented**: written in this repo from the paper. No upstream code was copied.

Licenses were read from the vendored LICENSE files, or from GitHub
(`api.github.com/repos/<repo>/license`) when no LICENSE file is in the tree
(checked 2026-10-08).

## World models

| project | upstream | license | in this repo | how | license file | our changes |
|---|---|---|---|---|---|---|
| Cosmos-Predict2 (DreamGen backbone, GR00T Dreams post-training) | [nvidia-cosmos/cosmos-predict2](https://github.com/nvidia-cosmos/cosmos-predict2); DreamGen pipeline: [NVIDIA/GR00T-Dreams](https://github.com/NVIDIA/GR00T-Dreams) | Apache-2.0 | `models/dreamgen/` | vendored (fork) | [`models/dreamgen/LICENSE`](../models/dreamgen/LICENSE); dependency notices: [`models/dreamgen/ATTRIBUTIONS.md`](../models/dreamgen/ATTRIBUTIONS.md) | embodiment configs (`cosmos_predict2/configs/base/experiment/`), the LVG / multiview rollout examples, and cache / sparse-attention / pruning hooks (`add_dreamgen_cache_args` etc.) |
| DreamDojo | [NVIDIA/DreamDojo](https://github.com/NVIDIA/DreamDojo) | Apache-2.0 | `models/DreamDojo/` (incl. `packages/cosmos-*`, `external/lam*`, `groot_dreams/`) | vendored (fork) | [`models/DreamDojo/LICENSE`](../models/DreamDojo/LICENSE) | experiment configs for our embodiments, method hooks (cache, sparse attention, pruning), attention profiler (`DD_ATTN_PROFILE`) |
| Ctrl-World | [Robert-gyj/Ctrl-World](https://github.com/Robert-gyj/Ctrl-World) | MIT | `models/Ctrl-World/` | vendored (fork) | [`models/Ctrl-World/LICENSE.txt`](../models/Ctrl-World/LICENSE.txt) | bimanual / humanoid / 4-view-grid configs (`config_*.py`), meta builders (`dataset_meta_info/`), method hooks. `models/pipeline_stable_video_diffusion.py` is derived from 🤗 diffusers (Apache-2.0, header kept) |
| openpi (π0.5 policy for Ctrl-World policy-in-the-loop) | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | Apache-2.0 | `models/Ctrl-World/openpi` @ `c23745b` | gitlink | `LICENSE` in the checkout | local edits in the cluster checkout only (`src/openpi/models/model.py`, `shared/download.py`, `training/data_loader.py`; not tracked here) |

Model weights are not stored in git. They follow their own licenses (Cosmos-Predict2 /
DreamDojo: NVIDIA Open Model License; Stable Video Diffusion: Stability AI license; T5:
Apache-2.0). See [checkpoints/CHECKPOINTS.md](../checkpoints/CHECKPOINTS.md).

## Acceleration methods

| method | upstream | license | in this repo | how | license file | our changes |
|---|---|---|---|---|---|---|
| FasterCache (Lv et al., ICLR 2025) | [Vchitect/FasterCache](https://github.com/Vchitect/FasterCache) | **none declared upstream** | `methods/cache_strategy/FasterCache/` (`fastercache/`, `taming/`, `scripts/`, `configs/`) | vendored | **missing**: upstream has no LICENSE. Nested LPIPS code keeps its BSD-2 files under `fastercache/models/cogvideox/sgm/modules/autoencoding/lpips/{loss,model}/LICENSE` | ours: `adapter.py`, `config.py`, `processor.py`, `runtime.py` (SVD / DiT FFT-delta + block-reuse port) |
| DiCache (Bu et al., ICLR 2026) | [Bujiazi/DiCache](https://github.com/Bujiazi/DiCache) | none declared upstream | `methods/cache_strategy/DiCache/` | re-implemented | n/a | written in this repo for Cosmos DiT, DreamDojo and the SVD U-Net |
| WorldCache (Nawaz et al., ECCV 2026) | [umair1221/WorldCache](https://github.com/umair1221/WorldCache) | none declared upstream | `methods/cache_strategy/WorldCache/` | re-implemented | n/a | written in this repo (motion-adaptive threshold + HF / saliency / OSI / flow guards) |
| Sparse VideoGen 1 / 2 (Xi et al.; Yang et al.) | [svg-project/Sparse-VideoGen](https://github.com/svg-project/Sparse-VideoGen) | Apache-2.0 | `methods/sparse_attention/Sparse-VideoGen/` | vendored (unmodified) | [`LICENSE.txt`](../methods/sparse_attention/Sparse-VideoGen/LICENSE.txt) | none. Adapters are ours: `methods/sparse_attention/svg/` |
| ↳ CUTLASS | [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) | BSD-3-Clause | `…/Sparse-VideoGen/svg/kernels/3rdparty/cutlass` @ `08185b9` | gitlink | `LICENSE.txt` in the checkout | none |
| ↳ FlashInfer | [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | Apache-2.0 | `…/3rdparty/flashinfer` @ `a1166dc` | gitlink | `LICENSE` in the checkout | none (the venvs use the `flashinfer-python` wheel) |
| ↳ pybind11 | [pybind/pybind11](https://github.com/pybind/pybind11) | BSD-3-Clause-style | `…/3rdparty/pybind` @ `2889136` | gitlink | `LICENSE` in the checkout | none |
| Radial Attention (Li et al.) | [mit-han-lab/radial-attention](https://github.com/mit-han-lab/radial-attention) | Apache-2.0 | `methods/sparse_attention/radial-attention/` | vendored | [`LICENCE.txt`](../methods/sparse_attention/radial-attention/LICENCE.txt) | none. Adapters are ours: `methods/sparse_attention/radial/` (incl. the fp16 → bf16 output cast fix) |
| PISA, Piecewise Sparse Attention (Li et al.) | [xie-lab-ml/piecewise-sparse-attention](https://github.com/xie-lab-ml/piecewise-sparse-attention) | MIT | `methods/sparse_attention/piecewise-sparse-attention/`; kernel copy `methods/sparse_attention/pisa/piecewise_sparse_attn_tma.py` | vendored | [`LICENSE`](../methods/sparse_attention/piecewise-sparse-attention/LICENSE) (covers the kernel copy, header kept) | adapters are ours: `methods/sparse_attention/pisa/` |
| SiTo (Zhang et al., AAAI 2025) | [EvelynZhang-epiclab/SiTo](https://github.com/EvelynZhang-epiclab/SiTo), itself built on [dbolya/tomesd](https://github.com/dbolya/tomesd) (MIT) | **none declared upstream** | `methods/prunning/SiTo/` (`sito.py`, `utils.py`, `patch.py`, `README.md` from upstream) | vendored + extended | **missing**: upstream has no LICENSE | ours: `adapter.py`, `config.py`, `processor.py`, `runtime.py`, `spatiotemporal_hold.py` (prune-once-hold), plan caching |
| ITM, Importance-Based Token Merging (Wu et al., ICCV 2025) | [cvlab-stonybrook/importance-token-merge](https://github.com/cvlab-stonybrook/importance-token-merge) (builds on tomesd) | MIT | `methods/prunning/importance-token-merge/` | vendored | [`LICENSE`](../methods/prunning/importance-token-merge/LICENSE) | adapters are ours: `methods/prunning/importance_token_merge/` (`block_hold.py`, CFG-free feature-L2 importance) |

## Metrics

| component | upstream | license | in this repo | how | license file | our changes |
|---|---|---|---|---|---|---|
| WorldScore (Duan et al., ICCV 2025): backends for motion amplitude / smoothness | [haoyi-duan/WorldScore](https://github.com/haoyi-duan/WorldScore) | MIT | `metrics/WorldScore` @ `344983c` | gitlink | `LICENSE` in the checkout | one local edit in the cluster checkout (`third_party/groundingdino/models/GroundingDINO/ms_deform_attn.py`; not tracked here). Loaded through `metrics/third_party.py` |
| ↳ SEA-RAFT (optical flow) | [princeton-vl/SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT) | BSD-3-Clause | `metrics/WorldScore/worldscore/benchmark/metrics/third_party/SEA-RAFT` | inside the WorldScore gitlink | none in the WorldScore copy | none |
| ↳ VFIMamba (frame interpolation) | [MCG-NJU/VFIMamba](https://github.com/MCG-NJU/VFIMamba) | Apache-2.0 | `…/third_party/VFIMamba` | inside the WorldScore gitlink | none in the WorldScore copy | none |
| ↳ GroundingDINO (subject boxes) | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | Apache-2.0 | `…/third_party/groundingdino` | inside the WorldScore gitlink | none in the WorldScore copy | see WorldScore row |
| ↳ SAM 2.1 (subject masks) | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | Apache-2.0 | `…/third_party/sam2` | inside the WorldScore gitlink | none in the WorldScore copy | none |
| Motion Amplitude Score (RBench, Deng et al., ICML 2026) | paper only | n/a | `metrics/motion_amplitude.py` | re-implemented | n/a | closed-world adaptation (camera-mount-aware statistic) |
| VGGT-Omega (RIGIS depth + camera backbone) | [facebookresearch/vggt-omega](https://github.com/facebookresearch/vggt-omega) | **FAIR Noncommercial Research License** | `metrics/rigis/vggt_omega/` | vendored | [`metrics/rigis/vggt_omega/LICENSE`](../metrics/rigis/vggt_omega/LICENSE) (added 2026-10-08; it was missing) | none. RIGIS scoring (`multiview_consistency/`, `run.sh`) is ours, see [`metrics/rigis/VENDORED.md`](../metrics/rigis/VENDORED.md) |
| KIVA (kinescore) | in-house (vendored from an internal branch) | MIT (`metrics/kiva/pyproject.toml`) | `metrics/kiva/` | vendored (ours) | declared in `pyproject.toml` | see [`metrics/kiva/VENDORED.md`](../metrics/kiva/VENDORED.md) |
| ↳ DINOv3 (KIVA reader backbone; weights `facebook/dinov3-vit{s,l}16-pretrain-lvd1689m`) | [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) | **DINOv3 License** | loaded through `transformers` | pip + HF weights | upstream | none |
| ↳ pytorch-kinematics | [UM-ARM-Lab/pytorch_kinematics](https://github.com/UM-ARM-Lab/pytorch_kinematics) | MIT | KIVA env | pip | upstream | none |
| ↳ robot_descriptions (URDF / MJCF loaders) | [robot-descriptions/robot_descriptions.py](https://github.com/robot-descriptions/robot_descriptions.py) | Apache-2.0 (each robot description has its own license) | KIVA env | pip | upstream | none |
| CLIP-IQA+ (Visual Integrity) | [chaofengc/IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch) (`pyiqa`) | **PolyForm Noncommercial 1.0.0** | `models/dreamgen/.venv` | pip | upstream | p10-over-frames aggregation in `metrics/visual_integrity.py` |
| CLIPScore (Text Alignment) | [Lightning-AI/torchmetrics](https://github.com/Lightning-AI/torchmetrics) + [openai/CLIP](https://github.com/openai/CLIP) ViT-B/16 | Apache-2.0 / MIT | `models/dreamgen/.venv` | pip | upstream | task-clause prompt, per-frame scoring (`metrics/text_alignment.py`) |
| PSNR / LPIPS / SSIM | [photosynthesis-team/piq](https://github.com/photosynthesis-team/piq) (LPIPS weights from [richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity), BSD-2-Clause); SSIM from torchmetrics | Apache-2.0 | `models/dreamgen/.venv` | pip | upstream | `metrics/fidelity.py` |
| FVD, I3D features | FVD: Unterthiner et al., 2018. I3D: [google-deepmind/kinetics-i3d](https://github.com/google-deepmind/kinetics-i3d) (Apache-2.0), used through the TorchScript export distributed with [universome/stylegan-v](https://github.com/universome/stylegan-v) | Apache-2.0 (I3D) | `metrics/fvd/` (ours); weights `checkpoints/metrics/fvd/i3d_torchscript.pt` | re-implemented + weights | n/a | shared real pool, 5 fps windows ([`metrics/fvd/README.md`](../metrics/fvd/README.md)) |

## Missing or incomplete license files

| path | status |
|---|---|
| `methods/cache_strategy/FasterCache/` | upstream repo declares no license; nothing to copy. Ask the authors before redistributing. |
| `methods/prunning/SiTo/` | upstream repo declares no license; nothing to copy. The ToMe-derived parts are MIT (tomesd). |
| `metrics/rigis/vggt_omega/` | **fixed 2026-10-08**: upstream `LICENSE` added. |
| `metrics/WorldScore/.../third_party/{SEA-RAFT,VFIMamba,groundingdino,sam2}` | the WorldScore copies carry no LICENSE files (upstream WorldScore issue); licenses are listed above. |
| repo root | RoboWorM's own code has no top-level `LICENSE` file yet. |

## Fetching the gitlinked checkouts

The gitlinks have no `.gitmodules` entries, so clone them by hand after cloning this repo.
Each command checks out the commit this repo records:

```bash
clone_at() { git clone "$1" "$2" && git -C "$2" checkout "$3"; }
clone_at https://github.com/haoyi-duan/WorldScore.git           metrics/WorldScore         344983c1be7d515c1e4ecb167d40d622adf502bc
clone_at https://github.com/Physical-Intelligence/openpi.git    models/Ctrl-World/openpi   c23745b5ad24e98f66967ea795a07b2588ed6c79
K=methods/sparse_attention/Sparse-VideoGen/svg/kernels/3rdparty
clone_at https://github.com/NVIDIA/cutlass.git                 $K/cutlass                 08185b9c3e90510ee2b656662ed0d53b06d28157
clone_at https://github.com/flashinfer-ai/flashinfer.git       $K/flashinfer              a1166dc0169b479aa3220b61759547d04c64e473
clone_at https://github.com/pybind/pybind11.git                $K/pybind                  288913638bb2da563f1c39e7d07071c2f21bfb25
```

`metrics/WorldScore` is needed for motion amplitude / smoothness. `openpi` is needed only
for the π0.5 policy-in-the-loop Ctrl-World launcher. The SVG kernel sources are needed
only to rebuild `svg/kernels` (the benchmark uses the flashinfer wheel).

## Datasets

Episodes come from DROID, RoboChallenge, AIST Bimanual, RoboCOIN and GR00T Teleop GR1,
and each keeps the license of its source corpus. See
[datasets/README.md](datasets/README.md).
