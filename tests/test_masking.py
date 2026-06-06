from __future__ import annotations

import numpy as np

from jepa_poc.data.masking import sample_masks


def test_masks_are_disjoint_and_roughly_correct_size():
    rng = np.random.default_rng(13)
    n_genes = 100
    for _ in range(1000):
        mask = sample_masks(n_genes, ctx_frac=0.3, n_target_blocks=2, target_frac=0.2, rng=rng)
        context = set(mask.context_idx.tolist())
        assert abs(len(context) - 30) <= 1
        seen = set(context)
        for block in mask.target_blocks:
            block_set = set(block.tolist())
            assert abs(len(block_set) - 20) <= 1
            assert seen.isdisjoint(block_set)
            seen |= block_set
