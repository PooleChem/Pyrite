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
) -> Poses:
    """Generate random placements and conformations of a molecule in a binding site.

    Used to create the initial poses of a search. Positions and orientations come from the
    `binding_site`, and the torsions from the conformations of `mol`; both are then combined.

    .. note::
        Only the euler rotation layout is supported for now (the `binding_site` samples euler
        angles), and the randomness comes from the global ``numpy.random`` state.

    Parameters
    ----------
    mol : Mol
        The molecule to place.
    binding_site : Bounds
        The binding site in which to place the molecule.
    n_positions : int
        The number of molecule positions to generate. When `placement` = ``grid``, this is the
        size of the grid in every axis. For example, an `n_positions` of 4 would yield
        :math:`4^3` positions.
    n_conformations : int
        The number of conformations to generate.
    placement : {'random', 'grid'}, default 'random'
        The placement method to use. ``random`` places the molecule randomly in the binding
        site. ``grid`` creates a grid in the binding site.
    conformations : {'conformer', 'random'}, default 'conformer'
        The conformer generation method to use. ``conformer`` will create conformers using
        RDKit :func:`~rdkit.Chem.rdDistGeom.EmbedMultipleConfs`. ``random`` will set
        all dihedral angles to random values. This is faster, but can create
        physically impossible configurations.
    combine : {'random', 'grid'}, default 'random'
        The combination method. ``random`` will create random combinations of positions and
        dihedral angles. When ``n_positions >= n_conformations``, a random conformation is
        chosen for every position, and the other way around. This results in an output size of
        ``max(n_positions, n_conformations)``.
        ``grid`` combines all positions with all conformations, resulting in an output size of
        ``n_positions * n_conformations``.

    Returns
    -------
    Poses
        Shape ``(max(n_positions, n_conformations), n_dims)`` when `combine` is ``random``, or
        ``(n_positions * n_conformations, n_dims)`` when `combine` is ``grid``, in the layout
        of `mol`.
    """
    if mol.layout.rot_type != "euler":
        raise NotImplementedError(
            f"place_in only supports the 'euler' rotation layout, not {mol.layout.rot_type!r}: the "
            "binding site samples euler angles."
        )
    if placement not in {"random", "grid"}:
        raise ValueError("placement must be either 'random' or 'grid'")
    if conformations not in {"conformer", "random"}:
        raise ValueError("conformations must be either 'conformer' or 'random'")
    if combine not in {"random", "grid"}:
        raise ValueError("combine must be either 'random' or 'grid'")

    if placement == "random":
        positions = binding_site.place_random_uniform(n_positions)
    else:
        positions = binding_site.place_grid(n_positions)

    if conformations == "conformer":
        dihedrals = mol.get_n_conformer_dihedral_configurations(n_conformations)
    else:
        dihedrals = mol.get_n_random_dihedral_configurations(n_conformations)

    if combine == "random":
        if positions.shape[0] >= dihedrals.shape[0]:
            sel_dihedrals = np.random.choice(
                dihedrals.shape[0], size=positions.shape[0], replace=True
            )
            out_pos = positions
            out_dih = dihedrals[sel_dihedrals]
        else:
            sel_positions = np.random.choice(
                positions.shape[0], size=dihedrals.shape[0], replace=True
            )
            out_pos = positions[sel_positions]
            out_dih = dihedrals
    else:
        out_pos = np.repeat(positions, dihedrals.shape[0], axis=0)
        out_dih = np.tile(dihedrals, (positions.shape[0], 1))

    return Poses(np.concatenate((out_pos, out_dih), axis=1), mol.layout)


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
