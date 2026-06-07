from __future__ import annotations

import torch
import torch.nn.functional as F

from methods.cache_strategy.common import WorldCacheConfig
from methods.cache_strategy.DiCache.ctrl_world_runtime import CtrlWorldDiCacheRuntime, _rel_l1


def _compute_hf_drift(prev_img: torch.Tensor, curr_img: torch.Tensor) -> torch.Tensor:
    channels = prev_img.shape[1]
    kernel = torch.tensor(
        [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
        dtype=prev_img.dtype,
        device=prev_img.device,
    ).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    prev_hf = F.conv2d(prev_img, kernel, padding=1, groups=channels)
    curr_hf = F.conv2d(curr_img, kernel, padding=1, groups=channels)
    return (prev_hf - curr_hf).abs().mean()


def _compute_saliency_map(features: torch.Tensor) -> torch.Tensor:
    saliency = torch.std(features, dim=1, keepdim=True)
    flat = saliency.reshape(saliency.shape[0], -1)
    mn = flat.min(dim=1, keepdim=True)[0].reshape(saliency.shape[0], 1, 1, 1)
    mx = flat.max(dim=1, keepdim=True)[0].reshape(saliency.shape[0], 1, 1, 1)
    return (saliency - mn) / (mx - mn + 1e-6)


def _compute_optimal_gamma(delta_curr: torch.Tensor, delta_prev: torch.Tensor) -> float:
    delta_curr_flat = delta_curr.reshape(1, -1)
    delta_prev_flat = delta_prev.reshape(1, -1)
    numer = (delta_curr_flat * delta_prev_flat).sum(dim=1, keepdim=True)
    denom = (delta_prev_flat * delta_prev_flat).sum(dim=1, keepdim=True)
    gamma = torch.clamp(numer / (denom + 1e-6), 0.2, 1.8)
    return float(gamma.item())


class CtrlWorldWorldCacheRuntime(CtrlWorldDiCacheRuntime):
    config: WorldCacheConfig

    def __init__(self, *, config: WorldCacheConfig, total_steps: int) -> None:
        super().__init__(config=config, total_steps=total_steps)
        self.config = config

    def _probe_metric(self, curr: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        metric = _rel_l1(curr, prev)
        if not getattr(self.config, "saliency_enabled", False):
            return metric

        diff_map = (curr - prev).abs().mean(dim=1, keepdim=True)
        saliency = _compute_saliency_map(prev)
        beta = float(getattr(self.config, "saliency_weight", 5.0))
        weighted_drift = (diff_map * (1.0 + beta * saliency)).mean()
        return weighted_drift / (prev.abs().mean() + 1e-6)

    def _skip_threshold(self) -> float | None:
        if self.probe_state_for_step is None or self.previous_probe_state is None:
            return None
        if self.probe_state_for_step.shape != self.previous_probe_state.shape:
            return None

        threshold = float(self.config.rel_l1_thresh)
        if self.previous_input is not None and self.step_input is not None:
            if self.step_input.shape != self.previous_input.shape:
                return None
            input_velocity = _rel_l1(self.step_input, self.previous_input)
            threshold = threshold / (1.0 + float(self.config.motion_sensitivity) * float(input_velocity.item()))

        if getattr(self.config, "dynamic_decay", False):
            total_steps = max(int(self.total_steps), 1)
            u_ratio = total_steps / 35.0
            base_mult = (u_ratio**2) / 6.0 + u_ratio / 2.0 + 10.0 / 3.0
            threshold *= 1.0 + base_mult * (self.current_step_idx / total_steps)

        if getattr(self.config, "hf_enabled", False) and self.previous_input is not None and self.step_input is not None:
            prev_img = self.previous_input.reshape(self.previous_input.shape[0], -1, *self.previous_input.shape[-2:])
            curr_img = self.step_input.reshape(self.step_input.shape[0], -1, *self.step_input.shape[-2:])
            hf_drift = _compute_hf_drift(prev_img, curr_img)
            if hf_drift > float(getattr(self.config, "hf_thresh", 0.01)):
                return None

        return super()._skip_threshold() if threshold == float(self.config.rel_l1_thresh) else self._skip_threshold_with_value(threshold)

    def _skip_threshold_with_value(self, threshold: float) -> float | None:
        if self.probe_state_for_step is None or self.previous_probe_state is None:
            return None
        if not self.do_classifier_free_guidance or self.batch_size < 2:
            return threshold if self._probe_metric(self.probe_state_for_step, self.previous_probe_state) < threshold else None

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
            delta_curr = self.probe_state_for_step - self.full_probe_window[-1]
            delta_prev = self.full_probe_window[-1] - self.full_probe_window[-2]
            if getattr(self.config, "osi_enabled", False):
                gamma = _compute_optimal_gamma(delta_curr, delta_prev)
            else:
                numer = (delta_curr - delta_prev).abs().mean()
                denom = delta_prev.abs().mean()
                gamma = 1.0 if denom <= 1e-6 else float(torch.clamp(numer / denom, 1.0, 2.0).item())
            return self.output_window[-2] + gamma * (self.output_window[-1] - self.output_window[-2])
        return super()._predict_output()
