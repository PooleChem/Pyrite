from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from pyrite._common import Poses


def boltzmann_diversity_filter(
    poses: Poses,
    energies: ArrayLike,
    k: int,
    T: float | None = None,
    pool_factor: int = 5,
    rng: np.random.Generator | None = None,
) -> Poses:
    """Select `k` low-energy poses that are spread out in space.

    Used to choose the starting poses of a search from a large set of cheaply scored
    placements. First a pool of ``pool_factor * k`` poses is drawn without replacement, with
    probability proportional to the Boltzmann weight ``exp(-(E - E_min) / T)``. From that pool,
    poses are then picked one by one: the first is the lowest-energy one, and every next one
    maximizes the distance of its translation to the closest pose picked so far, multiplied by
    its Boltzmann weight relative to the heaviest pose in the pool. This favours poses that are
    both good and far from those already chosen.

    Only the translation enters the distance, not the rotation or torsions.

    Parameters
    ----------
    poses : Poses
        The candidate poses.
    energies : array_like
        The score of each pose, shape ``(len(poses),)``. Lower is better.
    k : int
        The number of poses to select. If there are fewer poses than that, all are returned.
    T : float, optional
        The Boltzmann temperature, in the units of `energies`. Defaults to half the standard
        deviation of the energies (or uniform weights if they are all equal).
    pool_factor : int, default 5
        The size of the pool, in units of `k`.
    rng : numpy.random.Generator, optional
        The source of randomness. Defaults to a fresh generator.

    Returns
    -------
    Poses
        At most `k` poses, in the order they were picked. Fewer than `k` are returned if
        fewer than `k` poses have a Boltzmann weight that does not underflow to zero, which can
        happen with a small explicit `T`.
    """
    energies = np.asarray(energies, dtype=float)
    n = len(poses)
    if energies.shape != (n,):
        raise ValueError(f"energies must have shape ({n},), got {energies.shape}")
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if T is not None and T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    rng = np.random.default_rng() if rng is None else rng

    shifted = energies - energies.min()
    if T is None:
        T = float(shifted.std() * 0.5)
    weights = np.exp(-shifted / T) if T > 0 else np.ones(n)
    weights /= weights.sum()

    pool_size = min(pool_factor * k, np.count_nonzero(weights))
    k = min(k, pool_size)
    pool_idx = rng.choice(n, size=pool_size, replace=False, p=weights)
    pool = poses[pool_idx]
    pool_w = weights[pool_idx] / weights[pool_idx].max()
    positions = pool.translation

    mask = np.zeros(pool_size, dtype=bool)
    seed = int(np.argmin(energies[pool_idx]))
    selected = [seed]
    mask[seed] = True
    min_dists = np.linalg.norm(positions - positions[seed], axis=1)
    for _ in range(k - 1):
        scores = min_dists * pool_w
        scores[mask] = -np.inf
        ni = int(np.argmax(scores))
        selected.append(ni)
        mask[ni] = True
        min_dists = np.minimum(min_dists, np.linalg.norm(positions - positions[ni], axis=1))
    return pool[selected]
