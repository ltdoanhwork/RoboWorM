"""
PISA adapter for DreamGen (Cosmos Predict 2) architecture.

Injects Piecewise Sparse Attention into the DiT blocks by overriding the attn_op slot.
The wrapper implements the same interface as MinimalA2AAttnOp so that context-parallel
pipelines (set_context_parallel_group / disable_context_parallel) work correctly.
"""

import torch
import torch.nn as nn
import torch.distributed as dist

try:
    from .piecewise_sparse_attn_tma import piecewise_sparse_attention
    print("[PISA] Successfully loaded vendored piecewise_sparse_attn_tma kernel.")
except ImportError:
    msg = (
        "[PISA] Fatal: Could not import vendored piecewise_sparse_attn_tma kernel. "
        "PISA acceleration is required but unavailable in this workspace."
    )
    # Using a warning here but will raise error in the class if used
    print(msg)
    piecewise_sparse_attention = None


# ---------------------------------------------------------------------------
# PISAAttentionWrapper
# ---------------------------------------------------------------------------

class PISAAttentionWrapper(nn.Module):
    """
    Drop-in replacement for MinimalA2AAttnOp that requires PISA
    as the local attention kernel.

    Interface contract (matches MinimalA2AAttnOp / DistributedAttention):
      - forward(q, k, v, **kwargs) → Tensor  shape (B, S, H*D)
      - set_context_parallel_group(process_group, ranks, stream) → None
        (stores group/stream for A2A scatter-gather; called by Attention.set_context_parallel_group)

    Input q/k/v shape: (B, S, H, D)   ← same convention as Cosmos DiT
    Output shape:      (B, S, H*D)     ← same as torch_attention_op / MinimalA2AAttnOp
    """

    def __init__(self, density: float = 0.5, block_size: int = 64):
        super().__init__()
        self.density = density
        self.block_size = block_size
        # Context-parallel state — mirrors DistributedAttention fields
        self._cp_group = None
        self._cp_stream = None

    # ------------------------------------------------------------------
    # Context-parallel interface — required by Cosmos DiT pipeline
    # ------------------------------------------------------------------

    def set_context_parallel_group(self, process_group, ranks, stream) -> None:
        """
        Signature matches MinimalA2AAttnOp.set_context_parallel_group.
        Called by both enable_context_parallel and disable_context_parallel:
          - enable: process_group=<group>, ranks=[...], stream=<stream>
          - disable: process_group=None, ranks=[], stream=None
        """
        del ranks  # unused, same as MinimalA2AAttnOp
        self._cp_group = process_group
        self._cp_stream = stream

    # ------------------------------------------------------------------
    # Local attention kernel
    # ------------------------------------------------------------------

    def _local_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Runs PISA kernel on local q/k/v. Raises error if unavailable.
        Input:  (B, S, H, D)
        Output: (B, S, H, D)   ← un-flattened; caller reshapes to (B, S, H*D)
        """
        T = q.shape[1]

        if piecewise_sparse_attention is None:
            raise RuntimeError(
                "[PISA] piecewise_sparse_attention kernel is not loaded. "
                "Fallback to SDPA is disabled as requested."
            )

        if T < 128:
            raise RuntimeError(
                f"[PISA] Sequence length T={T} is below the enforced minimum T >= 128. "
                "Fallback is disabled."
            )

        # PISA expects (B, H, S, D)
        bs = min(self.block_size, T)
        out = piecewise_sparse_attention(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            density=self.density,
            block_size=bs,
        )
        return out.transpose(1, 2)   # (B, S, H, D)

    # ------------------------------------------------------------------
    # forward — handles single-GPU and context-parallel cases
    # ------------------------------------------------------------------

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            q, k, v: (B, S, H, D)
        Returns:
            (B, S, H*D)  — matches MinimalA2AAttnOp / torch_attention_op output
        """
        B, S, H, D = q.shape

        # ---- No context parallelism (default) ----
        if self._cp_group is None or dist.get_world_size(self._cp_group) < 2:
            out = self._local_attn(q, k, v)          # (B, S, H, D)
            return out.reshape(B, S, H * D)

        # ---- Context parallelism: A2A QKV scatter / gather ----
        from cosmos_predict2.module.a2a_cp import _SeqAllToAllQKV, _SeqAllToAll

        pg_size = dist.get_world_size(self._cp_group)

        # Scatter: (local-seq, global-head) → (global-seq, local-head)
        q_all, k_all, v_all = _SeqAllToAllQKV.apply(
            self._cp_group, q, k, v, pg_size, self._cp_stream, True
        )

        # Run local PISA on the full-sequence shard with local heads
        out_all = self._local_attn(q_all, k_all, v_all)   # (B, S_global, H_local, D)

        # Gather: (global-seq, local-head) → (local-seq, global-head)
        out_gathered = _SeqAllToAll.apply(self._cp_group, out_all, False)  # (B, S_local, H, D)

        B2, S_local, H2, D2 = out_gathered.shape
        return out_gathered.reshape(B2, S_local, H2 * D2)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def enable_pisa_dreamgen(
    model: nn.Module,
    density: float = 0.5,
    block_size: int = 64,
    start_layer_idx: int = 0,
) -> None:
    """
    Injects PISAAttentionWrapper into every self-attention block of a
    Cosmos Predict 2 / DreamGen DiT model at or after start_layer_idx.
    """
    if not hasattr(model, "blocks"):
        print("[PISA] Warning: Model does not have 'blocks' attribute. Injection failed.")
        return

    count = 0
    for i, block in enumerate(model.blocks):
        if i < start_layer_idx:
            continue
        if hasattr(block, "self_attn"):
            block.self_attn.attn_op = PISAAttentionWrapper(
                density=density, block_size=block_size
            )
            count += 1

    print(
        f"[PISA] Enabled on {count} blocks "
        f"(density={density}, block_size={block_size}, start_layer_idx={start_layer_idx})"
    )


def disable_pisa_dreamgen(model: nn.Module) -> None:
    """
    Restores default attention operators (MinimalA2AAttnOp) on all PISA-patched blocks.
    """
    from cosmos_predict2.module.a2a_cp import MinimalA2AAttnOp

    if not hasattr(model, "blocks"):
        return

    count = 0
    for block in model.blocks:
        if hasattr(block, "self_attn") and isinstance(
            block.self_attn.attn_op, PISAAttentionWrapper
        ):
            block.self_attn.attn_op = MinimalA2AAttnOp()
            count += 1

    print(f"[PISA] Disabled on {count} blocks — restored MinimalA2AAttnOp.")
