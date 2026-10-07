from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from pyrite._common import Mol, Poses

if TYPE_CHECKING:
    from pyrite.bounds import Bounds


def place_in(
    mol: Mol,
    binding_site: Bounds,
    n_positions: int,
    n_conformations: int,
    placement: str = "random",
    conformations: str = "conformer",
    combine: str = "random",
    rng: np.random.Generator | None = None,
) -> Poses:
    """Generate random placements and conformations of a molecule in a binding site.

    Used to create the initial poses of a search. A *placement* is a position, from the
    `binding_site`, together with a uniformly random orientation; the torsions come from
    conformations of `mol`. The two are then combined.

    Parameters
    ----------
    mol : Mol
        The molecule to place. The poses are in its layout.
    binding_site : Bounds
        The binding site in which to place the molecule.
    n_positions : int
        The number of placements to generate. When `placement` = ``grid``, this is the
        size of the grid in every axis. For example, an `n_positions` of 4 would yield
        :math:`4^3` placements.
    n_conformations : int
        The number of conformations to generate.
    placement : {'random', 'grid'}, default 'random'
        The placement method to use. ``random`` places the molecule randomly in the binding
        site. ``grid`` creates a grid of positions in the binding site, each with a random
        orientation.
    conformations : {'conformer', 'random'}, default 'conformer'
        The conformer generation method to use. ``conformer`` will create conformers using
        RDKit `EmbedMultipleConfs <https://www.rdkit.org/docs/source/rdkit.Chem.rdDistGeom.html>`_. ``random`` will set
        all torsion angles to random values, see
        :meth:`~pyrite.PoseLayout.sample_random_torsions`. This is faster, but can create
        physically impossible configurations.
    combine : {'random', 'grid'}, default 'random'
        The combination method. ``random`` will create random combinations of placements and
        torsions. When ``n_positions >= n_conformations``, a random conformation is
        chosen for every placement, and the other way around. This results in an output size of
        ``max(n_positions, n_conformations)``.
        ``grid`` combines all placements with all conformations, resulting in an output size of
        ``n_positions * n_conformations``.
    rng : numpy.random.Generator, optional
        The single source of randomness: the positions, the orientations, the conformer
        generation and the combination all use it, so one seed reproduces the result. Defaults
        to a fresh generator.

    Returns
    -------
    Poses
        Shape ``(max(n_positions, n_conformations), n_dims)`` when `combine` is ``random``, or
        ``(n_positions * n_conformations, n_dims)`` when `combine` is ``grid``, in the layout
        of `mol`. RDKit can return fewer conformers than asked for, in which case the number of
        conformations is smaller.

    See Also
    --------
    boltzmann_diversity_filter : Select starting poses from the placements.
    pyrite.Poses.from_parts : Assemble poses yourself.

    Examples
    --------
    >>> pocket = Pocket.from_mol(receptor).intersect(RectangularBounds.autobox(ligand, 1.0), padding=2.0)
    >>> placements = place_in(ligand, pocket, n_positions=2000, n_conformations=20, rng=np.random.default_rng(0))
    """
    if placement not in {"random", "grid"}:
        raise ValueError("placement must be either 'random' or 'grid'")
    if conformations not in {"conformer", "random"}:
        raise ValueError("conformations must be either 'conformer' or 'random'")
    if combine not in {"random", "grid"}:
        raise ValueError("combine must be either 'random' or 'grid'")
    rng = np.random.default_rng() if rng is None else rng
    layout = mol.layout

    if placement == "random":
        positions = binding_site.place_random_uniform(n_positions, rng=rng)
    else:
        positions = binding_site.place_grid(n_positions)
    rotations = layout.sample_random_rotations(len(positions), rng=rng)

    if conformations == "conformer":
        torsions = mol.get_n_conformer_torsion_configurations(
            n_conformations, seed=int(rng.integers(2**31 - 1))
        )
    else:
        torsions = layout.sample_random_torsions(n_conformations, rng=rng)

    n_placements, n_torsions = len(positions), len(torsions)
    if combine == "random":
        if n_placements >= n_torsions:
            out_rotations, out_positions = rotations, positions
            out_torsions = torsions[rng.choice(n_torsions, size=n_placements, replace=True)]
        else:
            chosen = rng.choice(n_placements, size=n_torsions, replace=True)
            out_rotations, out_positions = rotations[chosen], positions[chosen]
            out_torsions = torsions
    else:
        out_rotations = np.repeat(rotations, n_torsions, axis=0)
        out_positions = np.repeat(positions, n_torsions, axis=0)
        out_torsions = np.tile(torsions, (n_placements, 1))

    return Poses.from_parts(out_rotations, out_positions, out_torsions, layout=layout)


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

    See Also
    --------
    place_in : Make the placements to filter.

    Examples
    --------
    >>> energies = (DistanceToPocket(ligand, pocket) + InternalOverlap(ligand)).batch_scores(placements)
    >>> starts = boltzmann_diversity_filter(placements, energies, k=32)
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
