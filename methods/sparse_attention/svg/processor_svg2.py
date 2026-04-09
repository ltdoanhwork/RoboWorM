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
  4. Apply block-sparse attention (via flashinfer if available, else masked SDPA).
  5. Apply inverse permutation to restore original token order.

Dependencies:
  - svg.kmeans_utils: batch_kmeans_Euclid, identify_dynamic_map,
                      dynamic_block_sparse_fwd_flashinfer, density_calculation
  - svg.kernels.triton.permute: permute_tensor_by_labels_triton,
                                apply_inverse_permutation_triton
  - flashinfer: for dynamic_block_sparse_fwd_flashinfer

If flashinfer is not installed, SVG2 falls back to masked SDPA on a
permuted-then-padded tensor (correct but not as fast as the block-sparse kernel).
"""

from __future__ import annotations

import warnings
from typing import Optional

import torch
import torch.nn.functional as F

from .processor_svg1 import SVDSVGAttnProcessor, _split_heads

# ── lazy imports from SVG source (injected into sys.path by adapter.py) ──────

def _try_import_kmeans():
    try:
        from svg.kmeans_utils import (
            batch_kmeans_Euclid,
            identify_dynamic_map,
            dynamic_block_sparse_fwd_flashinfer,
            density_calculation,
        )
        return batch_kmeans_Euclid, identify_dynamic_map, \
               dynamic_block_sparse_fwd_flashinfer, density_calculation
    except ImportError as e:
        warnings.warn(
            f"[SVG2] svg.kmeans_utils not found: {e}. "
            "Install the SVG package: pip install -e methods/sparse_attention/Sparse-VideoGen"
        )
        return None, None, None, None


def _try_import_triton_permute():
    try:
        from svg.kernels.triton.permute import (
            permute_tensor_by_labels_triton,
            apply_inverse_permutation_triton,
        )
        return permute_tensor_by_labels_triton, apply_inverse_permutation_triton
    except ImportError as e:
        warnings.warn(
            f"[SVG2] svg.kernels.triton.permute not found: {e}. "
            "Build SVG kernels: cd methods/sparse_attention/Sparse-VideoGen && bash svg/kernels/setup.sh"
        )
        return None, None


class SVDSAPAttnProcessor(SVDSVGAttnProcessor):
    """
    SVG2 Semantic Aware Permutation processor for SVD UNet.

    Inherits fallback and head-splitting logic from SVDSVGAttnProcessor.
    Overrides the core sparse attention with K-Means + block sparse.

    Class-level attributes set by adapter.py:
        num_q_centroids:   int  — K-Means clusters for queries.
        num_k_centroids:   int  — K-Means clusters for keys.
        top_p_kmeans:      float — top-p threshold for block selection.
        min_kc_ratio:      float — minimum fraction of key blocks per query block.
        kmeans_iter_init:  int  — K-Means iterations on first call.
        kmeans_iter_step:  int  — K-Means iterations on subsequent calls.
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

        # Lazy-load SVG utilities
        (
            self._batch_kmeans,
            self._identify_dynamic_map,
            self._block_sparse_fwd,
            self._density_calc,
        ) = _try_import_kmeans()
        self._permute_fn, self._inv_permute_fn = _try_import_triton_permute()

        self._svg2_available = (
            self._batch_kmeans is not None and self._permute_fn is not None
        )
        if not self._svg2_available:
            warnings.warn(
                "[SVG2] Required imports not available. "
                "SVG2 will fall back to full SDPA until SVG package + kernels are installed."
            )

    # ── override core sparse attention ────────────────────────────────────
    def _svg1_attention(self, q, k, v, T):
        """Overrides SVG1 logic with SVG2 SAP when available."""
        if not self._svg2_available:
            # Pure SDPA fallback
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
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

        try:
            # 1. Run K-Means on Q and K
            qlabels, qcentroids, qcluster_sizes = self._run_kmeans(q, k)

            # 2. Build dynamic sparse map
            q_szs = qcluster_sizes[0].view(B, H, self.num_q_centroids)
            k_szs = qcluster_sizes[1].view(B, H, self.num_k_centroids)
            dynamic_map = self._identify_dynamic_map(
                qcentroids.view(B, H, self.num_q_centroids, D_h),
                qcentroids[1].view(B, H, self.num_k_centroids, D_h),   # k centroids
                q_szs, k_szs,
                self.top_p_kmeans, self.min_kc_ratio,
            )

            # 3. Permute Q, K, V by cluster labels
            q_perm, q_idx = self._permute_fn(q, qlabels[0], dim=2)
            k_perm, k_idx = self._permute_fn(k, qlabels[1], dim=2)
            v_perm, _     = self._permute_fn(v, qlabels[1], dim=2, sorted_indices=k_idx)

            # 4. Block-sparse attention
            if self._block_sparse_fwd is not None:
                try:
                    out_perm = self._block_sparse_fwd(
                        q_perm, k_perm, v_perm, dynamic_map, q_szs, k_szs, is_cpu=False
                    )
                except Exception as e:
                    warnings.warn(f"[SVG2] block_sparse_fwd failed ({e}), falling back to SDPA.")
                    out_perm = F.scaled_dot_product_attention(q_perm, k_perm, v_perm, dropout_p=0.0)
            else:
                out_perm = F.scaled_dot_product_attention(q_perm, k_perm, v_perm, dropout_p=0.0)

            # 5. Inverse permutation
            out = self._inv_permute_fn(out_perm, q_idx.unsqueeze(0), dim=2)
            return out.reshape(B, H, T, D_h)

        except Exception as e:
            # Any failure → safe SDPA fallback
            warnings.warn(f"[SVG2] SAP attention failed ({e}), falling back to full SDPA.")
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

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

        # Package for caller: labels as (qlabels, klabels), centroids as tuple,
        # cluster sizes as tuple
        return (qlabels, klabels), (qcentroids, kcentroids), (qcs, kcs)
