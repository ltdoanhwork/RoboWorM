# tmp/ — traceable archive (not active code)

`tmp/` is git-ignored. The exceptions are this README and the `*.md` files in `tmp/20261005_refactor/`.
Nothing in the active tree (`scripts/`, `metrics/`, `models/`, ...) should import or call anything under `tmp/`.

## `tmp/20261005_refactor/`

This folder is the archive of the **2026-10-05 paper-aligned refactor**. That refactor cut the repo down to what the RoboWorM
(ICLR 2027) paper uses. Anything taken out of the active tree was **moved** here, not deleted, so it can be traced
and restored.

- [`20261005_refactor/MANIFEST.md`](20261005_refactor/MANIFEST.md) has one row per moved item: original path, new path, size,
  reason and restore command.
- [`20261005_refactor/REGENERATE_REPORTS.md`](20261005_refactor/REGENERATE_REPORTS.md) covers the `metrics/reports/` files that were lost
  from disk on 2026-10-05. It lists which files were recovered and the script that regenerates each missing one.
- Subfolders: `web/` (webpage, labeling and web-staging), `supabase/`, `vendor/` (e.g. libgl), `scripts/` (dead or one-off
  scripts), `docs/` (finished hand-off notes), `checkpoints_unused/` and `misc/`.

The snapshot commit `WIP snapshot before paper-aligned refactor` on branch `refactor/paper-cleanup` holds the
tree as it was before any move.

## Other contents

- `tmp/wc_ladder/` is scratch output from the WorldCache break-point ladder
  (`scripts/_make_worldcache_ladder.sh`, humanoid single/multiview rungs). It is not part of the refactor, and it is safe to
  delete once the numbers in `.cursor/rules/dreamgen-humanoid-singleview-caching-insights.mdc` are no longer needed.
