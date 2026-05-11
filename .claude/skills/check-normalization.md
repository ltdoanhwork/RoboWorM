---
name: check-normalization
description: Verify normalization statistics and transforms for a training config before launching
user_invocable: true
---

# Check Normalization

When invoked, verify that the normalization pipeline is correctly configured for the specified embodiment/training config. This skill MUST be triggered before any training launch.

## Steps

1. **Identify the embodiment** from the YAML config or user request (aloha, droid, gr1, g1, yam, agibot).

2. **Check shared_meta stats file** at `models/DreamDojo/shared_meta/<EMBODIMENT>_stats.json`:
   - Verify `observation.state` and `action` keys exist with `mean`, `std`, `min`, `max`, `q01`, `q99`
   - Verify array lengths match the expected dimensionality:
     - ALOHA: 14-dim (state.qpos, action.qpos)
     - DROID: 7-dim (6 cartesian + 1 gripper for both state and action)
     - GR1: 29-dim
     - G1: 43-dim
     - YAM: 46-dim
     - AgiBot: 22-dim
   - Check that stats are NOT placeholder (all zeros mean / all ones std is suspicious)
   - Compare against a sample dataset's `meta/stats.json` to verify consistency

3. **Check shared_meta modality file** at `models/DreamDojo/shared_meta/<EMBODIMENT>_modality.json`:
   - Verify `start` and `end` indices match the expected action dim
   - Verify `original_key` names match what the dataset actually has in its parquet columns

4. **Check the transform pipeline** in `models/DreamDojo/groot_dreams/groot_configs.py`:
   - Confirm `StateActionTransform` uses correct `normalization_modes` (typically `"min_max"`)
   - Confirm all action/state modality keys are included in the transform

5. **Check action slicing** in `models/DreamDojo/groot_dreams/data/dataset.py` `__getitem__`:
   - Verify the embodiment's action range is correct:
     - GR1: `action_seq[:, 0:29]`
     - G1: `action_seq[:, 58:101]`
     - YAM: `action_seq[:, 101:147]`
     - AgiBot: `action_seq[:, 147:169]`
     - ALOHA: `action_seq[:, 169:183]`
     - DROID: `action_seq[:, 183:190]`
   - Verify the path-matching condition correctly identifies the dataset

6. **Validate with a sample** (if dataset exists):
   ```python
   python3 -c "
   import json, numpy as np
   from pathlib import Path
   # Load one dataset's stats
   ds_path = '<first_dataset_path_from_yaml>/meta/stats.json'
   with open(ds_path) as f: stats = json.load(f)
   # Load default stats
   default_path = 'models/DreamDojo/shared_meta/<EMBODIMENT>_stats.json'
   with open(default_path) as f: default = json.load(f)
   # Check dimension match
   action_key = 'action'  # or 'observation.state'
   print(f'Dataset action dim: {len(stats[action_key][\"mean\"])}')
   print(f'Default action dim: {len(default[action_key][\"mean\"])}')
   # Check range sanity
   ds_min = np.array(stats[action_key]['min'])
   ds_max = np.array(stats[action_key]['max'])
   default_min = np.array(default[action_key]['min'])
   default_max = np.array(default[action_key]['max'])
   print(f'Default covers dataset range: {np.all(default_min <= ds_min) and np.all(default_max >= ds_max)}')
   "
   ```

7. **Report** a summary:
   - Stats file: OK / PLACEHOLDER / MISSING
   - Modality file: OK / MISMATCH
   - Transform mode: min_max / mean_std / q99
   - Action slice: correct range / WRONG
   - Sample validation: PASS / FAIL (with details)

## When to trigger

- Before ANY `bash launch.sh` or `torchrun` command
- When adding a new embodiment
- When changing dataset paths in YAML configs
- When modifying `groot_configs.py` or `dataset.py` action slicing
