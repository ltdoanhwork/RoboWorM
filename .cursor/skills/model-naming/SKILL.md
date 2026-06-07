---
name: model-naming
description: Naming conventions for DreamGen vs DreamDojo models, scripts, and directories. Use when creating or renaming scripts, referencing models, or resolving confusion between the two systems.
---

# Model Naming Convention: DreamGen vs DreamDojo

## Two Separate Systems

| | DreamGen | DreamDojo |
|---|---|---|
| **Code** | `models/dreamgen/` (cosmos_predict2 pipeline) | `models/DreamDojo/` (action-conditioned) |
| **Training** | `finetuning/dreamgen/` | Separate (exp_aloha_resume70k) |
| **Logs** | `exp_logs/dreamgen_logs/` | `exp_logs/dreamdojo_logs/` |
| **Venv** | `models/DreamDojo/.venv` (shared) | `models/DreamDojo/.venv` |
| **Inference module** | `models.dreamgen.examples.video2world` | `models.DreamDojo.examples.action_conditioned` |

## Script Naming Rules

**Use `dreamgen` when:**
- Script imports from `models.dreamgen.*`
- Script uses checkpoints from `exp_logs/dreamgen_logs/`
- Script sets `PYTHONPATH` to `models/dreamgen`

**Use `dreamdojo` when:**
- Script imports from `models.DreamDojo.*`
- Script uses checkpoints from `exp_logs/dreamdojo_logs/`
- Script sets `PYTHONPATH` to `models/DreamDojo`

## Script Name Pattern

```
run_{embodiment}_{view}_{model}_{variant}.sh
```

- `{embodiment}`: `bimanual`, `humanoid`, `single_arm`
- `{view}`: `singleview`, `multiview`
- `{model}`: `dreamgen` or `dreamdojo` (match the actual codebase used)
- `{variant}`: `dense`, `full_episode`, `worldcache`, `fastercache`, `dicache`, `itm`, etc.

## Quick Check

If unsure which name to use, check inside the script:
```bash
grep -l "models\.dreamgen\|PYTHONPATH.*models/dreamgen" script.sh  # → name it dreamgen
grep -l "models\.DreamDojo\|PYTHONPATH.*models/DreamDojo" script.sh  # → name it dreamdojo
```
