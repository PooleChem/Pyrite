"""Clustering of poses by RMSD, and selection of a representative of every cluster."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from rdkit.ML.Cluster import Butina

from pyrite._common import Mol, Poses
from pyrite._util import _heavy_atoms, _symmetry_mappings


def rmsd_matrix(
    mol: Mol,
    poses: Poses,
    symmetry: bool = True,
    max_matches: int = 1000,
) -> NDArray:
    """Calculate the RMSD between every pair of poses of a molecule.

    The poses are compared in place, so they are *not* aligned onto each other: this is the RMSD
    that tells how far a docked pose lies from another one. Only the heavy atoms are used, so the
    hydrogens `mol` has, which depend on how it was loaded, do not change it.

    The memory needed is ``O(n_poses ** 2)``.

    Parameters
    ----------
    mol : Mol
        The molecule the poses belong to. It is not modified.
    poses : Poses
        The poses, in the layout of `mol`.
    symmetry : bool, default True
        Whether to take the symmetry of the molecule into account. Atoms that are equivalent in
        the molecular graph (the two ortho carbons of a phenyl ring, the oxygens of a carboxylate,
        ...) can be swapped, and the lowest RMSD over all swaps is used. Without it, a pose and
        the same pose with a flipped ring are far apart.
    max_matches : int, default 1000
        The maximum number of symmetry-equivalent atom mappings to consider. A molecule with many
        symmetric groups has exponentially many; a warning is issued when this limit is reached,
        as the RMSD can then be overestimated. The identity mapping is always included, so the
        result is never larger than without symmetry.

    Returns
    -------
    numpy.ndarray
        A symmetric array of shape ``(n_poses, n_poses)`` with the RMSD in Angstrom, and zeros on
        the diagonal.
    """
    positions = mol.pose_to_positions(poses)
    n, n_atoms, _ = positions.shape
    if n == 0:
        return np.zeros((0, 0))
    if not np.isfinite(positions).all():
        raise ValueError("The poses contain non-finite atom positions.")

    heavy = _heavy_atoms(mol)
    if symmetry:
        matches = _symmetry_mappings(mol, mol, max_matches, include_identity=True, heavy_atoms=True)
    else:
        matches = [tuple(heavy)]
    positions, n_atoms = positions[:, heavy], len(heavy)
    matches = [np.searchsorted(heavy, m) for m in matches]  # into the heavy-atom positions

    # The RMSD is a Euclidean distance between flattened positions, so with a permutation sigma of
    # the atoms, d^2(i, j) = |x_i|^2 + |x_j|^2 - 2 x_i . x_j[sigma]. A shift common to all poses does
    # not change it, but keeps the squares small, which limits the cancellation in this expansion.
    positions = positions - positions.mean(axis=(0, 1))
    x = positions.reshape(n, -1)
    sq = np.einsum("ij,ij->i", x, x)
    d2 = np.full((n, n), np.inf)
    # The positions are finite (checked above), so the floating-point flags NumPy raises for the
    # matrix product are spurious (seen with Apple's Accelerate BLAS, also for plain `a @ a.T`).
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for sigma in matches:
            x_sigma = positions[:, list(sigma), :].reshape(n, -1)
            np.minimum(d2, sq[:, None] + sq[None, :] - 2.0 * (x @ x_sigma.T), out=d2)

    rmsd = np.sqrt(np.maximum(d2, 0.0) / n_atoms)
    rmsd = 0.5 * (rmsd + rmsd.T)
    np.fill_diagonal(rmsd, 0.0)
    return rmsd


def cluster_and_select(
    mol: Mol,
    poses: Poses,
    scores: ArrayLike,
    cutoff: float = 2.0,
    n_output: int | None = None,
    symmetry: bool = True,
    max_matches: int = 1000,
) -> NDArray:
    """Cluster poses by RMSD and select the best-scoring pose of every cluster.

    The poses are clustered with the Butina algorithm: poses closer than `cutoff` to each other
    are neighbours, and clusters are built around the poses with the most neighbours. From every
    cluster the pose with the lowest score is selected.

    The selected poses are not necessarily further apart than `cutoff` from each other, as the
    cluster centre is not necessarily the best-scoring member.

    Parameters
    ----------
    mol : Mol
        The molecule the poses belong to. It is not modified.
    poses : Poses
        The poses, in the layout of `mol`.
    scores : array_like
        The score of every pose, shape ``(len(poses),)``. Lower is better.
    cutoff : float, default 2.0
        The RMSD in Angstrom up to which two poses are neighbours.
    n_output : int, optional
        The maximum number of poses to return. By default one per cluster is returned.
    symmetry : bool, default True
        Whether to take the symmetry of the molecule into account in the RMSD, see
        :func:`rmsd_matrix`.
    max_matches : int, default 1000
        The maximum number of symmetry-equivalent atom mappings, see :func:`rmsd_matrix`.

    Returns
    -------
    numpy.ndarray
        The indices into `poses` of the selected poses, ordered from the lowest to the highest
        score. Use ``poses[indices]`` and ``scores[indices]`` to get them.

    Examples
    --------
    >>> indices = cluster_and_select(ligand, poses, scores, cutoff=2.0, n_output=10)
    >>> best = poses[indices]
    """
    scores = np.asarray(scores, dtype=float)
    n = len(poses)
    if scores.shape != (n,):
        raise ValueError(f"scores must have shape ({n},), got {scores.shape}")
    if n_output is not None and n_output < 1:
        raise ValueError(f"n_output must be at least 1, got {n_output}")
    if n == 0:
        return np.empty(0, dtype=np.intp)

    distances = rmsd_matrix(mol, poses, symmetry=symmetry, max_matches=max_matches)
    clusters = Butina.ClusterData(distances, n, cutoff, isDistData=True, reordering=True)
    best = np.array([min(cluster, key=lambda i: scores[i]) for cluster in clusters], dtype=np.intp)
    return best[np.argsort(scores[best], kind="stable")][:n_output]
