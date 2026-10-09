import argparse
from typing import Any

from methods.cache_strategy.common import (
    BSCConfig,
    BWCacheConfig,
    CacheRuntimeConfig,
    DiCacheConfig,
    FasterCacheConfig,
    ScalingCacheConfig,
    RFCacheConfig,
    WorldCacheConfig,
)


def add_dreamgen_cache_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("DreamGen Cache Backends")
    
    # FasterCache
    group.add_argument("--fastercache-enabled", action="store_true", help="Enable FasterCache acceleration.")
    group.add_argument("--fastercache-start-step", type=int, default=0, help="Step index after which caching may kick in.")
    group.add_argument("--fastercache-model-interval", type=int, default=5, help="Reuse cached whole-model output every N steps.")
    group.add_argument("--fastercache-block-interval", type=int, default=3, help="Reuse cached block attention output every N steps.")
    group.add_argument("--fastercache-debug", action="store_true", help="Enable debug logs for FasterCache.")
    
    # WorldCache
    group.add_argument("--worldcache-enabled", action="store_true", help="Enable WorldCache acceleration for DiT inference.")
    group.add_argument("--worldcache-num-steps", type=int, default=35, help="Total denoising steps (must match --num-steps in inference).")
    group.add_argument("--worldcache-rel-l1-thresh", type=float, default=0.03, help="Relative L1 threshold for cache hit decision.")
    group.add_argument("--worldcache-ret-ratio", type=float, default=0.4, help="Fraction of initial steps to always compute (warm-up).")
    group.add_argument("--worldcache-probe-depth", type=int, default=4, help="Number of leading DiT blocks used as the probe.")
    group.add_argument("--worldcache-motion-sensitivity", type=float, default=5.0, help="Alpha in dynamic threshold.")
    group.add_argument("--worldcache-flow-enabled", action="store_true", help="Warp cached features using optical-flow before reuse.")
    group.add_argument("--worldcache-flow-scale", type=float, default=0.5, help="Down-scale factor for flow estimation (0-1).")
    group.add_argument("--worldcache-hf-enabled", action="store_true", help="Enable spectral (high-frequency) drift guard.")
    group.add_argument("--worldcache-hf-thresh", type=float, default=0.01, help="HF drift above this forces a full compute step.")
    group.add_argument("--worldcache-saliency-enabled", action="store_true", help="Weight drift by channel-variance saliency map.")
    group.add_argument("--worldcache-saliency-weight", type=float, default=5.0, help="Beta multiplier for saliency-guided thresholding.")
    group.add_argument("--worldcache-osi-enabled", action="store_true", help="Online System Identification for optimal gamma.")
    group.add_argument("--worldcache-dynamic-decay", action="store_true", help="Increase threshold over time.")
    group.add_argument("--worldcache-aduc-enabled", action="store_true", help="Adaptive Unconditional Caching.")
    group.add_argument("--worldcache-aduc-start", type=float, default=0.5, help="Step ratio after which AdUC activates.")
    group.add_argument("--worldcache-parallel-cfg", action="store_true", help="Batch cond+uncond in B=2.")

    # DiCache
    group.add_argument("--dicache-enabled", action="store_true", help="Enable DiCache acceleration.")
    group.add_argument("--dicache-num-steps", type=int, default=35, help="Denoising steps.")
    group.add_argument("--dicache-rel-l1-thresh", type=float, default=0.08, help="Relative threshold.")
    group.add_argument("--dicache-ret-ratio", type=float, default=0.2, help="Retention ratio.")
    group.add_argument("--dicache-probe-depth", type=int, default=2, help="Probe depth.")

    # ScalingCache (implemented in AM_DiT/src/scl)
    group.add_argument("--scalingcache-enabled", action="store_true", help="Enable ScalingCache (difference-scaling + dynamic-interval) from AM_DiT/src/scl.")
    group.add_argument("--scalingcache-num-steps", type=int, default=35, help="Total denoising steps (must match inference --num-steps).")
    group.add_argument("--scalingcache-first-enhance", type=int, default=10, help="Warm-up steps always fully computed.")
    group.add_argument("--scalingcache-last-enhance", type=int, default=2, help="Final steps always fully computed.")
    group.add_argument("--scalingcache-error-rate", type=float, default=1.0, help="Scales the adaptive error threshold (dynamic mode).")
    group.add_argument("--scalingcache-fresh-threshold", type=int, default=2, help="Fixed refresh period when --scalingcache-no-dynamic.")
    group.add_argument("--scalingcache-no-dynamic", action="store_true", help="Use fixed periodic refresh instead of error-driven scheduling.")
    group.add_argument("--scalingcache-use-alpha", action="store_true", help="Use offline-fitted scaling coefficients.")
    group.add_argument("--scalingcache-alpha-dict-path", type=str, default=None, help="Path to alpha_dict_*.pth (when --scalingcache-use-alpha).")
    group.add_argument("--scalingcache-update-alpha", action="store_true", help="Offline alpha-fitting pass (force Scaling, fit alphas).")
    group.add_argument("--scalingcache-granularity", type=str, default="block", choices=["block", "submodule"], help="Cache granularity: 'block' (memory-safe, default) or 'submodule' (faithful, ~3x memory).")
    group.add_argument("--scalingcache-no-warm-calibrate", dest="scalingcache_warm_calibrate", action="store_false", help="Disable Algorithm-1 seeding of delta_s (reproduces the unseeded defect; AM_DiT INSIGHTS SEC 61).")
    group.set_defaults(scalingcache_warm_calibrate=True)

    # BWCache (implemented in AM_DiT: methods/cache_strategy/BWCache/{dit,dreamgen}.py).
    # Units are DENOISE steps. Paper defaults: delta=0.15, R=10% of T, last_step=1/2.
    group.add_argument("--bwcache-enabled", action="store_true", help="Enable BWCache (step-level residual reuse) from AM_DiT.")
    group.add_argument("--bwcache-num-steps", type=int, default=35, help="Total denoising steps (overwritten per chunk by the pipeline's own count).")
    group.add_argument("--bwcache-rel-l1-thresh", type=float, default=0.15, help="Similarity threshold delta that triggers the periodic reuse schedule.")
    group.add_argument("--bwcache-reuse-interval", type=int, default=3, help="R: reuse R steps then compute 1.")
    group.add_argument("--bwcache-last-step", type=float, default=0.5, help="Fraction (<1) or count (>=1) of final steps always computed.")
    group.add_argument("--bwcache-warmup-steps", type=int, default=3, help="Leading denoise steps always computed.")
    # RFC (Son et al., ICLR 2026; implemented in AM_DiT: methods/cache_strategy/RFC). Units are
    # DENOISE steps. TaylorSeer module cache + RFE estimate + RCS schedule (full compute once
    # the accumulated first-module input rel-L1 forecast error exceeds tau).
    group.add_argument("--rfcache-enabled", action="store_true", help="Enable RFC (relational feature caching, ICLR 2026) from AM_DiT.")
    group.add_argument("--rfcache-num-steps", type=int, default=35, help="Total denoising steps (overwritten per chunk by the pipeline's own count).")
    group.add_argument("--rfcache-tau", type=float, default=0.7, help="RCS tau on the accumulated first-module input rel-L1 error.")
    group.add_argument("--rfcache-max-order", type=int, default=1, help="m: Taylor order (RFE direction + RCS input forecast).")
    group.add_argument("--rfcache-first-enhance", type=int, default=3, help="Leading denoise steps always computed.")
    group.add_argument("--rfcache-max-interval", type=int, default=0, help="Cap on consecutive cached steps (0 = paper, no cap).")
    group.add_argument("--rfcache-modules", type=str, default="self_attn,cross_attn,mlp", help="Comma list of block sub-modules to cache.")
    # BSC -- ours (implemented in AM_DiT: src/bsc core + methods/cache_strategy/BSC).
    # Unlike the step-level baselines above, BSC gates GROUPS OF DiT BLOCKS and predicts a
    # skipped group's output from its cached anchor, so the knobs are per-block/per-group.
    # Defaults are the Ctrl-World reference point from AM_DiT/docs/ALGORITHM.md section 9;
    # DreamGen is a different backbone and needs its own sweep.
    #
    # NOTE ON STEP COUNTS: DreamGen runs CFG sequentially, so net.forward (and therefore
    # the BSC step counter) fires TWICE per denoise step. --bsc-warmup-steps and
    # --bsc-min-full-interval are in net-forward units: 6 means "re-anchor every 3 denoise
    # steps". Same doubling reset_worldcache_state applies to its own num_steps.
    group.add_argument("--bsc-enabled", action="store_true", help="Enable BSC (block/group-level skip-eligibility caching) from AM_DiT.")
    group.add_argument("--bsc-gate-mode", type=str, default="group", choices=["group", "perblock_pooled", "step3", "step", "drift"], help="Gate granularity. 'group' (default) = one shared decision per contiguous block group.")
    group.add_argument("--bsc-rcs-threshold", type=float, default=0.1, help="tau: drift above which the group is fully computed. THE main speed/quality knob.")
    group.add_argument("--bsc-copy-threshold", type=float, default=0.03, help="Drift below this uses the free 'copy' predictor.")
    group.add_argument("--bsc-min-full-interval", type=int, default=3, help="Max reuse horizon: force a full compute at least every N net forwards (2 per denoise step).")
    group.add_argument("--bsc-warmup-steps", type=int, default=3, help="Initial net forwards always fully computed.")
    group.add_argument("--bsc-no-output-gate", dest="bsc_output_gate", action="store_false", help="Gate on the raw input drift E_I instead of the predicted output error s*E_I.")
    group.set_defaults(bsc_output_gate=True)
    # groups
    group.add_argument("--bsc-group-size", type=int, default=4, help="Contiguous DiT blocks per skip-group (1 = per-block).")
    group.add_argument("--bsc-group-predict", type=str, default="io", choices=["io", "block", "io_vel"], help="'io' = intermediate members reuse their cached output; 'block' = each predicts.")
    group.add_argument("--bsc-group-bounds", type=str, default=None, help="Non-uniform groups as 'a-b,c-d,...' (inclusive block indices). Overrides --bsc-group-size.")
    # protection
    group.add_argument("--bsc-protect-first-n", type=int, default=0, help="Force full compute for groups covering the first N blocks (the unstable head).")
    group.add_argument("--bsc-protect-last-frac", type=float, default=0.0, help="Force full compute during the final fraction of the denoise schedule.")
    group.add_argument("--bsc-protect-from-block", type=int, default=None, help="Always fully compute blocks with index >= this.")
    group.add_argument("--bsc-max-skip-run", type=int, default=0, help="Cap consecutive skipped blocks within a step's chain (0 = no cap).")
    # skip-eligibility
    group.add_argument("--bsc-skip-eligibility", action="store_true", help="Data-driven protection: force a group full when its measured I->O ratio is unreliable.")
    group.add_argument("--bsc-elig-s-dev", type=float, default=0.5, help="Eligibility: max |s_ema - 1| before a group is forced full.")
    group.add_argument("--bsc-elig-cv", type=float, default=0.35, help="Eligibility: max CV(s) before a group is forced full.")
    group.add_argument("--bsc-elig-warmup-fits", type=int, default=2, help="Full computes needed to seed the eligibility statistics.")
    group.add_argument("--bsc-s-ema-beta", type=float, default=0.8, help="EMA factor for the per-block I->O ratio.")
    # predictor
    group.add_argument("--bsc-predictor-mode", type=str, default="linear_s1", help="Skip predictor: linear_s1|copy|linear|linear_ema|hybrid_cos|linear_decay|block_extrap|...")
    group.add_argument("--bsc-s-clamp", type=float, default=None, help="Upper clamp on the fitted ratio s.")
    group.add_argument("--bsc-s-floor", type=float, default=None, help="Lower clamp on the fitted ratio s.")
    group.add_argument("--bsc-decay-rate", type=float, default=0.5, help="(linear_decay*) shrink the correction as the anchor goes stale.")
    # adaptive cadence
    group.add_argument("--bsc-adaptive-interval", action="store_true", help="Self-calibrating per-block re-anchor cadence instead of the fixed timer.")
    group.add_argument("--bsc-adaptive-rate", type=float, default=1.0, help="(--bsc-adaptive-interval) higher = skip more aggressively.")
    group.add_argument("--bsc-error-budget", type=float, default=None, help="Cumulative reuse-error budget; force a full step once exceeded.")
    group.add_argument("--bsc-mfi-max", type=int, default=12, help="Hard cap on the per-block effective min_full_interval.")
    group.add_argument("--bsc-cv-interval", type=float, default=0.0, help="Shorten the reuse horizon for volatile (high-CV) blocks.")
    group.add_argument("--bsc-depth-interval-scale", type=float, default=0.0, help="Lengthen the reuse horizon for mid-network blocks.")
    group.add_argument("--bsc-step-copy-threshold", type=float, default=0.0, help="Whole-step copy escalation threshold (0 = off).")
    # SNR-phase gate
    group.add_argument("--bsc-snr-skip-frac", type=float, default=0.0, help="Allow skipping only once the schedule fraction passes this (step-index proxy).")
    group.add_argument("--bsc-snr-skip-t", type=float, default=0.0, help="Physical noise gate: skip only when the true t=sigma/(sigma+1) is BELOW this. 0 = off.")


#: Every backend's selector flag, in the order `build_cache_runtime_config` tests them.
#: Kept as data so the mutual-exclusion check and the dispatch cannot disagree, and so the
#: single-GPU guards in `models/dreamgen/examples/video2world*.py` have one list to read
#: instead of each hard-coding its own (which is how BSC would have been quietly omitted).
CACHE_SELECTOR_FLAGS: tuple[str, ...] = (
    "fastercache_enabled",
    "worldcache_enabled",
    "dicache_enabled",
    "scalingcache_enabled",
    "bwcache_enabled",
    "rfcache_enabled",
    "bsc_enabled",
)


def enabled_cache_backends(setup_args: Any) -> list[str]:
    """Names of the cache backends selected on the command line."""
    names = [f.removesuffix("_enabled") for f in CACHE_SELECTOR_FLAGS
             if getattr(setup_args, f, False)]
    # Legacy alias kept for older launchers.
    if getattr(setup_args, "use_fastercache", False) and "fastercache" not in names:
        names.append("fastercache")
    return names


def build_cache_runtime_config(setup_args: Any) -> CacheRuntimeConfig | None:
    # Two backends at once would silently produce a number attributable to neither: each
    # patches the same DiT, and whichever ran second wraps the first's patched forwards.
    # Fail here, in the one place every entry point goes through, rather than relying on
    # each launcher to remember its own guard.
    selected = enabled_cache_backends(setup_args)
    if len(selected) > 1:
        raise ValueError(
            "[Cache] cache backends are mutually exclusive; got "
            f"{', '.join(sorted(selected))}. They all patch the same DiT, so enabling two "
            "yields a measurement attributable to neither. Pick one.")

    if getattr(setup_args, "fastercache_enabled", False) or getattr(setup_args, "use_fastercache", False):
        from methods.cache_strategy.common import FasterCacheConfig
        return FasterCacheConfig(
            start_step=getattr(setup_args, "fastercache_start_step", 0),
            model_interval=getattr(setup_args, "fastercache_model_interval", 5),
            block_interval=getattr(setup_args, "fastercache_block_interval", 3),
            debug=getattr(setup_args, "fastercache_debug", False),
        )
    if getattr(setup_args, "worldcache_enabled", False):
        return WorldCacheConfig(
            num_steps=getattr(setup_args, "worldcache_num_steps", 35),
            rel_l1_thresh=getattr(setup_args, "worldcache_rel_l1_thresh", 0.03),
            ret_ratio=getattr(setup_args, "worldcache_ret_ratio", 0.4),
            probe_depth=getattr(setup_args, "worldcache_probe_depth", 4),
            motion_sensitivity=getattr(setup_args, "worldcache_motion_sensitivity", 5.0),
            flow_enabled=getattr(setup_args, "worldcache_flow_enabled", False),
            flow_scale=getattr(setup_args, "worldcache_flow_scale", 0.5),
            hf_enabled=getattr(setup_args, "worldcache_hf_enabled", False),
            hf_thresh=getattr(setup_args, "worldcache_hf_thresh", 0.01),
            saliency_enabled=getattr(setup_args, "worldcache_saliency_enabled", False),
            saliency_weight=getattr(setup_args, "worldcache_saliency_weight", 5.0),
            osi_enabled=getattr(setup_args, "worldcache_osi_enabled", False),
            dynamic_decay=getattr(setup_args, "worldcache_dynamic_decay", False),
            aduc_enabled=getattr(setup_args, "worldcache_aduc_enabled", False),
            aduc_start=getattr(setup_args, "worldcache_aduc_start", 0.5),
            parallel_cfg=getattr(setup_args, "worldcache_parallel_cfg", False),
        )
    if getattr(setup_args, "dicache_enabled", False):
        return DiCacheConfig(
            num_steps=getattr(setup_args, "dicache_num_steps", 35),
            rel_l1_thresh=getattr(setup_args, "dicache_rel_l1_thresh", 0.08),
            ret_ratio=getattr(setup_args, "dicache_ret_ratio", 0.2),
            probe_depth=getattr(setup_args, "dicache_probe_depth", 2),
        )
    if getattr(setup_args, "scalingcache_enabled", False):
        return ScalingCacheConfig(
            num_steps=getattr(setup_args, "scalingcache_num_steps", 35),
            first_enhance=getattr(setup_args, "scalingcache_first_enhance", 10),
            last_enhance=getattr(setup_args, "scalingcache_last_enhance", 2),
            error_rate=getattr(setup_args, "scalingcache_error_rate", 1.0),
            fresh_threshold=getattr(setup_args, "scalingcache_fresh_threshold", 2),
            dynamic_cache=not getattr(setup_args, "scalingcache_no_dynamic", False),
            use_alpha=getattr(setup_args, "scalingcache_use_alpha", False),
            alpha_dict_path=getattr(setup_args, "scalingcache_alpha_dict_path", None),
            update_alpha=getattr(setup_args, "scalingcache_update_alpha", False),
            granularity=getattr(setup_args, "scalingcache_granularity", "block"),
            warm_calibrate=getattr(setup_args, "scalingcache_warm_calibrate", True),
        )
    if getattr(setup_args, "rfcache_enabled", False):
        return RFCacheConfig(
            num_steps=getattr(setup_args, "rfcache_num_steps", 35),
            tau=getattr(setup_args, "rfcache_tau", 0.7),
            max_order=getattr(setup_args, "rfcache_max_order", 1),
            first_enhance=getattr(setup_args, "rfcache_first_enhance", 3),
            max_interval=getattr(setup_args, "rfcache_max_interval", 0),
            modules=getattr(setup_args, "rfcache_modules", "self_attn,cross_attn,mlp"),
        )
    if getattr(setup_args, "bwcache_enabled", False):
        return BWCacheConfig(
            num_steps=getattr(setup_args, "bwcache_num_steps", 35),
            rel_l1_thresh=getattr(setup_args, "bwcache_rel_l1_thresh", 0.15),
            reuse_interval=getattr(setup_args, "bwcache_reuse_interval", 3),
            last_step=getattr(setup_args, "bwcache_last_step", 0.5),
            warmup_steps=getattr(setup_args, "bwcache_warmup_steps", 3),
        )
    if getattr(setup_args, "bsc_enabled", False):
        return BSCConfig(
            gate_mode=getattr(setup_args, "bsc_gate_mode", "group"),
            rcs_threshold=getattr(setup_args, "bsc_rcs_threshold", 0.1),
            copy_threshold=getattr(setup_args, "bsc_copy_threshold", 0.03),
            min_full_interval=getattr(setup_args, "bsc_min_full_interval", 3),
            warmup_steps=getattr(setup_args, "bsc_warmup_steps", 3),
            output_gate=getattr(setup_args, "bsc_output_gate", True),
            group_size=getattr(setup_args, "bsc_group_size", 4),
            group_predict=getattr(setup_args, "bsc_group_predict", "io"),
            group_bounds=getattr(setup_args, "bsc_group_bounds", None),
            protect_first_n=getattr(setup_args, "bsc_protect_first_n", 0),
            protect_last_frac=getattr(setup_args, "bsc_protect_last_frac", 0.0),
            protect_from_block=getattr(setup_args, "bsc_protect_from_block", None),
            max_skip_run=getattr(setup_args, "bsc_max_skip_run", 0),
            skip_eligibility=getattr(setup_args, "bsc_skip_eligibility", False),
            elig_s_dev=getattr(setup_args, "bsc_elig_s_dev", 0.5),
            elig_cv=getattr(setup_args, "bsc_elig_cv", 0.35),
            elig_warmup_fits=getattr(setup_args, "bsc_elig_warmup_fits", 2),
            s_ema_beta=getattr(setup_args, "bsc_s_ema_beta", 0.8),
            predictor_mode=getattr(setup_args, "bsc_predictor_mode", "linear_s1"),
            s_clamp=getattr(setup_args, "bsc_s_clamp", None),
            s_floor=getattr(setup_args, "bsc_s_floor", None),
            decay_rate=getattr(setup_args, "bsc_decay_rate", 0.5),
            adaptive_interval=getattr(setup_args, "bsc_adaptive_interval", False),
            adaptive_rate=getattr(setup_args, "bsc_adaptive_rate", 1.0),
            error_budget=getattr(setup_args, "bsc_error_budget", None),
            mfi_max=getattr(setup_args, "bsc_mfi_max", 12),
            cv_interval=getattr(setup_args, "bsc_cv_interval", 0.0),
            depth_interval_scale=getattr(setup_args, "bsc_depth_interval_scale", 0.0),
            step_copy_threshold=getattr(setup_args, "bsc_step_copy_threshold", 0.0),
            snr_skip_frac=getattr(setup_args, "bsc_snr_skip_frac", 0.0),
            snr_skip_t=getattr(setup_args, "bsc_snr_skip_t", 0.0),
        )
    return None
