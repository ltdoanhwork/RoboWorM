from __future__ import annotations

from dataclasses import dataclass, field

import torch

from methods.cache_strategy.common import DiCacheConfig


def _rel_l1(curr: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
    return (curr - prev).abs().mean() / (prev.abs().mean() + 1e-8)


def _compute_gamma(delta_curr: torch.Tensor, delta_prev: torch.Tensor) -> float:
    numer = (delta_curr - delta_prev).abs().mean()
    denom = delta_prev.abs().mean()
    if denom <= 1e-6:
        return 1.0
    gamma = torch.clamp(numer / denom, 1.0, 2.0)
    return float(gamma.item())


@dataclass
class CtrlWorldDiCacheRuntime:
    config: DiCacheConfig
    total_steps: int
    current_step_idx: int = -1
    do_classifier_free_guidance: bool = False
    step_input: torch.Tensor | None = None
    batch_size: int = 0
    num_frames: int = 0
    probe_stage_idx: int = 0
    probe_state_for_step: torch.Tensor | None = None
    previous_probe_state: torch.Tensor | None = None
    previous_input: torch.Tensor | None = None
    output_cache: torch.Tensor | None = None
    output_window: list[torch.Tensor] = field(default_factory=list)
    full_probe_window: list[torch.Tensor] = field(default_factory=list)
    step_skipped_count: int = 0

    def reset(self, total_steps: int) -> None:
        self.total_steps = int(total_steps)
        self.current_step_idx = -1
        self.do_classifier_free_guidance = False
        self.step_input = None
        self.batch_size = 0
        self.num_frames = 0
        self.probe_stage_idx = 0
        self.probe_state_for_step = None
        self.previous_probe_state = None
        self.previous_input = None
        self.output_cache = None
        self.output_window = []
        self.full_probe_window = []
        self.step_skipped_count = 0

    def begin_step(
        self,
        sample: torch.Tensor,
        *,
        step_idx: int,
        do_classifier_free_guidance: bool,
    ) -> None:
        self.current_step_idx = int(step_idx)
        self.do_classifier_free_guidance = bool(do_classifier_free_guidance)
        self.step_input = sample.detach().clone()
        self.batch_size = sample.shape[0]
        self.num_frames = sample.shape[1]
        self.probe_stage_idx = 0
        self.probe_state_for_step = None

    def observe_probe(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        self.probe_stage_idx += 1
        if self.probe_stage_idx < max(int(self.config.probe_depth), 1):
            return None
        self.probe_state_for_step = hidden_states.detach().clone()
        if not self._is_skip_eligible():
            return None
        if self._skip_threshold() is None:
            return None

        self.step_skipped_count += 1
        self.previous_probe_state = self.probe_state_for_step
        self.previous_input = self.step_input
        return self._predict_output()

    def finish_step(self, output: torch.Tensor) -> torch.Tensor:
        output = output.detach().clone()
        if self.probe_state_for_step is not None:
            self.previous_probe_state = self.probe_state_for_step
            self.full_probe_window.append(self.probe_state_for_step)
            if len(self.full_probe_window) > 2:
                del self.full_probe_window[:-2]

        self.output_cache = output
        self.output_window.append(output)
        if len(self.output_window) > 2:
            del self.output_window[:-2]

        self.previous_input = self.step_input
        return output

    def _is_skip_eligible(self) -> bool:
        if self.output_cache is None or self.previous_probe_state is None:
            return False
        eligible_from = int(self.total_steps * float(self.config.ret_ratio))
        return self.current_step_idx >= eligible_from

    def _split_point(self, tensor: torch.Tensor) -> int:
        return (self.batch_size // 2) * self.num_frames if self.do_classifier_free_guidance else tensor.shape[0]

    def _probe_metric(self, curr: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        return _rel_l1(curr, prev)

    def _skip_threshold(self) -> float | None:
        if self.probe_state_for_step is None or self.previous_probe_state is None:
            return None
        threshold = float(self.config.rel_l1_thresh)
        if not self.do_classifier_free_guidance or self.batch_size < 2:
            metric = self._probe_metric(self.probe_state_for_step, self.previous_probe_state)
            return threshold if metric < threshold else None

        split = self._split_point(self.probe_state_for_step)
        cond_metric = self._probe_metric(
            self.probe_state_for_step[split:],
            self.previous_probe_state[split:],
        )
        uncond_metric = self._probe_metric(
            self.probe_state_for_step[:split],
            self.previous_probe_state[:split],
        )
        if cond_metric < threshold and uncond_metric < threshold:
            return threshold
        return None

    def _predict_output(self) -> torch.Tensor:
        if len(self.output_window) >= 2 and len(self.full_probe_window) >= 2 and self.probe_state_for_step is not None:
            gamma = _compute_gamma(
                self.probe_state_for_step - self.full_probe_window[-1],
                self.full_probe_window[-1] - self.full_probe_window[-2],
            )
            return self.output_window[-2] + gamma * (self.output_window[-1] - self.output_window[-2])
        if self.output_cache is None:
            raise RuntimeError("[DiCache] Ctrl-World skip requested before output cache was initialized.")
        return self.output_cache
