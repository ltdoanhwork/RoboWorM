"""
SVG2 / SAP (Semantic Aware Permutation) Attention Processor for Ctrl-World SVD UNet.

Ported from WanAttn_SAPAttn_Processor (Sparse-VideoGen/svg/models/wan/attention.py)
with the same SVD adaptations as processor_svg1.py (no joint text tokens,
no timestep arg, FlexAttention → masked SDPA fallback).

Algorithm:
  1. Run K-Means on Q tokens → qlabels, qcentroids
     Run K-Means on K tokens → klabels, kcentroids
  2. Build a dynamic sparse map: which (q-cluster, k-cluster) pairs to compute.
  3. Permute Q, K, V by their cluster labels to make the sparse structure
     block-contiguous in memory.
  4. Apply block-sparse attention (via flashinfer if available).
  5. Apply inverse permutation to restore original token order.

Dependencies:
  - svg.kmeans_utils: batch_kmeans_Euclid, identify_dynamic_map,
                      dynamic_block_sparse_fwd_flashinfer, density_calculation
  - svg.kernels.triton.permute: permute_tensor_by_labels_triton,
                                apply_inverse_permutation_triton
  - flashinfer: for dynamic_block_sparse_fwd_flashinfer

Fallback: Disabled per user request ("tắt hết fallback"). This script will
CRASH if any dependency is missing or if any step fails.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional

from .processor_svg1 import SVDSVGAttnProcessor, _split_heads

# Mandatory imports from SVG source (path added by adapter.py)
from svg.kmeans_utils import (
    batch_kmeans_Euclid,
    identify_dynamic_map,
    dynamic_block_sparse_fwd_flashinfer,
    density_calculation,
)
from svg.kernels.triton.permute import (
    permute_tensor_by_labels_triton,
    apply_inverse_permutation_triton,
)

class SVDSAPAttnProcessor(SVDSVGAttnProcessor):
    """
    SVG2 Semantic Aware Permutation processor for SVD UNet.
    Hard-coded to use SVG2 logic; no silent fallback to full attention.
    """

    # ── SVG2-specific class-level state ───────────────────────────────────
    num_q_centroids: int = 50
    num_k_centroids: int = 200
    top_p_kmeans: float = 0.9
    min_kc_ratio: float = 0.0
    kmeans_iter_init: int = 5
    kmeans_iter_step: int = 2
    logging_file: Optional[str] = None

    def __init__(self, layer_idx: int, num_layers: int):
        super().__init__(layer_idx=layer_idx, num_layers=num_layers)
        # Per-instance K-Means state
        self._centroids_init: bool = False
        self._q_centroids = None
        self._k_centroids = None

        # Assign mandatory functions
        self._batch_kmeans = batch_kmeans_Euclid
        self._identify_dynamic_map = identify_dynamic_map
        self._block_sparse_fwd = dynamic_block_sparse_fwd_flashinfer
        self._density_calc = density_calculation
        self._permute_fn = permute_tensor_by_labels_triton
        self._inv_permute_fn = apply_inverse_permutation_triton

    # ── override core sparse attention ────────────────────────────────────
    def _svg1_attention(self, q, k, v, T):
        """Overrides SVG1 logic with SVG2 SAP."""
        return self._sap_attention(q, k, v, T)

    def _sap_attention(
        self,
        q: torch.Tensor,   # (B, H, T, D_h)
        k: torch.Tensor,
        v: torch.Tensor,
        T: int,
    ) -> torch.Tensor:
        """
        Full SVG2 SAP pipeline:
          kmeans → dynamic map → permute → block sparse → inverse permute.
        """
        B, H, _, D_h = q.shape

        # 1. Run K-Means on Q and K
        qlabels, qcentroids, qcluster_sizes = self._run_kmeans(q, k)

        # 2. Build dynamic sparse map
        q_szs = qcluster_sizes[0].view(B, H, self.num_q_centroids)
        k_szs = qcluster_sizes[1].view(B, H, self.num_k_centroids)
        dynamic_map = self._identify_dynamic_map(
            qcentroids[0].view(B, H, self.num_q_centroids, D_h),
            qcentroids[1].view(B, H, self.num_k_centroids, D_h),
            q_szs, k_szs,
            self.top_p_kmeans, self.min_kc_ratio,
        )

        # 3. Permute Q, K, V by cluster labels
        q_perm, q_idx = self._permute_fn(q, qlabels[0], dim=2)
        k_perm, k_idx = self._permute_fn(k, qlabels[1], dim=2)
        v_perm, _     = self._permute_fn(v, qlabels[1], dim=2, sorted_indices=k_idx)

        # 4. Block-sparse attention
        # No fallback here — if flashinfer is missing, it will crash.
        out_perm = self._block_sparse_fwd(
            q_perm, k_perm, v_perm, dynamic_map, q_szs, k_szs, is_cpu=False
        )

        # 5. Inverse permutation
        # q_idx shape is (B*H, T), which matches what the kernel expects for a flattened input.
        out = self._inv_permute_fn(out_perm, q_idx, dim=2)
        return out.reshape(B, H, T, D_h)

    # ── K-Means helpers ────────────────────────────────────────────────────
    def _run_kmeans(self, q, k):
        """
        Run K-Means on query and key tensors.
        Returns (qlabels, kcentroids_tuple, cluster_sizes_tuple).
        """
        B, H, T, D_h = q.shape

        max_iters = self.kmeans_iter_step if self._centroids_init else self.kmeans_iter_init
        init_q = self._q_centroids if self._centroids_init else None
        init_k = self._k_centroids if self._centroids_init else None

        qlabels, qcentroids, qcs, _ = self._batch_kmeans(
            q.reshape(B * H, T, D_h),
            n_clusters=self.num_q_centroids,
            max_iters=max_iters,
            init_centroids=init_q,
        )
        klabels, kcentroids, kcs, _ = self._batch_kmeans(
            k.reshape(B * H, T, D_h),
            n_clusters=self.num_k_centroids,
            max_iters=max_iters,
            init_centroids=init_k,
        )

        # Cache centroids for warm-start on subsequent steps
        self._q_centroids = qcentroids
        self._k_centroids = kcentroids
        self._centroids_init = True

        return (qlabels, klabels), (qcentroids, kcentroids), (qcs, kcs)
