from __future__ import annotations

import numpy as np

from utils.seg_raster.stage_s3d import (
    density_strata, density_stratified_derangement, permute_rasters)


def test_permutation_has_no_self_match_and_preserves_density_multiset() -> None:
    ratios = [0.01, 0.011, 0.02, 0.021, 0.08, 0.081, 0.2, 0.21]
    first = density_stratified_derangement(ratios, seed=20260827)
    second = density_stratified_derangement(ratios, seed=20260827)
    assert first == second
    assert sorted(first) == list(range(len(ratios)))
    assert all(index != donor for index, donor in enumerate(first))
    assert sum(density_strata(ratios)[index] != density_strata(ratios)[donor]
               for index, donor in enumerate(first)) <= 2
    raster = np.asarray(ratios, dtype=np.float32).reshape(-1, 1, 1, 1)
    permuted = permute_rasters(raster, first)
    assert np.allclose(sorted(permuted.reshape(-1).tolist()), sorted(ratios))
