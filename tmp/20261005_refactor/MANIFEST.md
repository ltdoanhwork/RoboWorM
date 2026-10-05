# 2026-10-05 refactor: move manifest

Every item moved out of the active tree during the paper-aligned refactor gets one row here.
Items are moved with `mv` and never deleted. The only exceptions are known junk files: `nohup.out`, `.DS_Store`, an empty `dist/`, `tests/` and `.codex-worktrees/`.
Paths are relative to the repo root. To restore an item, run its restore command from the repo root.

Subfolders: `web/`, `supabase/`, `vendor/`, `scripts/`, `docs/`, `checkpoints_unused/`, `misc/`.
Lost or regenerable `metrics/reports` files are listed in [REGENERATE_REPORTS.md](REGENERATE_REPORTS.md).

| original path | new path | size | reason | restore command |
|---|---|---|---|---|
