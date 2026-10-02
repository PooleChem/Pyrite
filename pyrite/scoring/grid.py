import numpy as np
from numba import njit

from pyrite.bounds import Bounds
from pyrite.scoring import Clamp, ScoringFunction
from pyrite.scoring._base import _CombinedScoringFunction, _ScaledScoringFunction
from pyrite.scoring.dependencies import Dependency, KNNDependency, PositionDependency


@njit
def _trilinear_sum(positions, values, columns, origin, step):
    """Sum, per pose, the trilinear interpolation of every atom in its own column of `values`.

    Parameters
    ----------
    positions : ndarray
        Shape ``(n_poses, n_atoms, 3)``.
    values : ndarray
        Shape ``(nx, ny, nz, n_columns)``, on a regular grid starting at `origin`.
    columns : ndarray
        Shape ``(n_atoms,)``, the column of `values` that every atom reads.
    origin, step : ndarray
        Shape ``(3,)``: the first grid vertex, and the (uniform) spacing, per axis.

    Returns
    -------
    ndarray
        Shape ``(n_poses,)``. Atoms outside the grid contribute 0.
    """
    n_poses, n_atoms, _ = positions.shape
    nx, ny, nz = values.shape[0], values.shape[1], values.shape[2]
    out = np.zeros(n_poses)
    for i in range(n_poses):
        total = 0.0
        for j in range(n_atoms):
            f0 = (positions[i, j, 0] - origin[0]) / step[0]
            f1 = (positions[i, j, 1] - origin[1]) / step[1]
            f2 = (positions[i, j, 2] - origin[2]) / step[2]
            if f0 < 0 or f1 < 0 or f2 < 0 or f0 > nx - 1 or f1 > ny - 1 or f2 > nz - 1:
                continue
            i0, i1, i2 = min(int(f0), nx - 2), min(int(f1), ny - 2), min(int(f2), nz - 2)
            d0, d1, d2 = f0 - i0, f1 - i1, f2 - i2
            c = columns[j]
            total += (
                values[i0, i1, i2, c] * (1 - d0) * (1 - d1) * (1 - d2)
                + values[i0 + 1, i1, i2, c] * d0 * (1 - d1) * (1 - d2)
                + values[i0, i1 + 1, i2, c] * (1 - d0) * d1 * (1 - d2)
                + values[i0, i1, i2 + 1, c] * (1 - d0) * (1 - d1) * d2
                + values[i0 + 1, i1 + 1, i2, c] * d0 * d1 * (1 - d2)
                + values[i0 + 1, i1, i2 + 1, c] * d0 * (1 - d1) * d2
                + values[i0, i1 + 1, i2 + 1, c] * (1 - d0) * d1 * d2
                + values[i0 + 1, i1 + 1, i2 + 1, c] * d0 * d1 * d2
            )
        out[i] = total
    return out


class GridScore(ScoringFunction):
    """
    Approximates a KNN-based scoring function with a precomputed 3D grid,
    replacing live neighbor search + kernel evaluation with trilinear
    interpolation during a search.

    Only meaningful for scoring functions built entirely from KNN-based
    terms — every leaf in `scoring_function`'s tree must implement
    `_score_field()` (the `_KNNScoringFunction` subclasses in `protein.py`
    do; terms that depend on the ligand's own internal conformation, like
    `InternalEnergy` or `NumTors`, don't and can't meaningfully be
    grid-approximated). Combine those separately, after the fact:
    ``grid_score(ligand, receptor) + 1e-2 * InternalEnergy(ligand)``.

    .. note::
        The grid's spatial extent is `binding_site`'s translation bounds,
        padded by the reference ligand's own maximum reach from its center
        atom (independent of rotation), plus `padding`. This is cheaper than
        the prototype notebook's approach (sampling many actual placements
        and taking the empirical envelope) but covers the same thing it was
        trying to guarantee — a ligand can extend well outside a bounding
        box that only constrains its *center*, once rotated.

    Parameters
    ----------
    scoring_function : ScoringFunction
        The (KNN-based) scoring function to approximate.
    binding_site : Bounds
        The region the ligand will actually be searched within — the grid
        must cover this, not wherever the ligand's conformer happens to sit
        when `GridScore` is constructed.
    spacing : float, default 0.5
        Grid spacing, in the same units as atomic coordinates (Angstrom).
    padding : float, default 4.0
        Extra padding added on top of the binding site + ligand-reach extent.
    """

    def __init__(
        self,
        scoring_function: ScoringFunction,
        binding_site: Bounds,
        spacing: float = 0.5,
        padding: float = 4.0,
    ):
        leaves = self._leaves(scoring_function)
        # getattr(...) is None, not hasattr — terms that can't support this (LJ,
        # ElectroStatic, AD4Solvation, PlantsPLP) explicitly set _score_field = None
        # to shadow the inherited-but-broken _KNNScoringFunction version, rather than
        # having it silently exist and crash with NotImplementedError when called.
        unsupported = [leaf for leaf in leaves if getattr(leaf, "_score_field", None) is None]
        if unsupported:
            names = ", ".join(sorted({type(leaf).__name__ for leaf in unsupported}))
            raise TypeError(
                f"Can't build a GridScore — these terms don't support grid "
                f"approximation (no _score_field() method): {names}"
            )

        self.scoring_function = scoring_function
        self.spacing = spacing
        self.padding = padding

        # any leaf works as the reference — they all operate on the same
        # bound ligand by construction (that's what makes combining them
        # into one composite meaningful in the first place)
        reference = leaves[0]
        self._probe_mol = reference.probe_mol
        self._probe_mask = reference.probe_mask
        self._position_dep = PositionDependency(self._probe_mol)

        merged = Dependency.merge_all(scoring_function.get_dependencies())
        knn_dep = next(iter(merged))

        axes, shape, grid_points = self._make_grid_points(binding_site, spacing, padding)

        # same KNNDependency machinery every other scoring function already
        # goes through — just point its query at grid vertices instead of
        # the real ligand. Reuses the already-built/cached KDTree (same
        # point_cloud -> same tree_hash -> KDTreeCache hit).
        grid_dep = KNNDependency(
            knn_dep.point_cloud,
            lambda _: grid_points,
            knn_dep.k,
            knn_dep.distance_upper_bound,
        )
        r_grid, idx_grid, _ = grid_dep.compute(None)

        # One column per atom type present in the ligand; every atom reads the column of its type.
        probe_types = self._probe_mol.atom_types[self._probe_mask]
        atom_types = sorted(set(probe_types.tolist()))
        self._values = np.ascontiguousarray(
            np.stack(
                [
                    scoring_function._score_field(r_grid, idx_grid, atom_type).reshape(shape)
                    for atom_type in atom_types
                ],
                axis=-1,
            )
        )
        self._columns = np.array([atom_types.index(t) for t in probe_types.tolist()])
        self._origin = np.array([axis[0] for axis in axes])
        self._step = np.array([axis[1] - axis[0] for axis in axes])

    def _make_grid_points(self, binding_site: Bounds, spacing: float, padding: float):
        # how far this ligand can reach from wherever its center atom ends up,
        # independent of rotation — covers the same gap the prototype's
        # sampled-envelope approach was closing, without needing to sample
        center = self._probe_mol.positions[self._probe_mol.center_atom]
        reach = np.linalg.norm(self._probe_mol.positions[self._probe_mask] - center, axis=1).max()

        translation_bounds = binding_site.get_translation_bounds()
        mins = np.array([lo for lo, _ in translation_bounds]) - reach - padding
        maxs = np.array([hi for _, hi in translation_bounds]) + reach + padding

        axes = [np.arange(lo, hi + spacing, spacing) for lo, hi in zip(mins, maxs)]
        shape = tuple(len(axis) for axis in axes)
        grid_points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        return axes, shape, grid_points

    @staticmethod
    def _leaves(sf: ScoringFunction) -> list[ScoringFunction]:
        """Walk a composite down to its actual leaf scoring functions."""
        if isinstance(sf, _CombinedScoringFunction):
            return [leaf for f in sf.funcs for leaf in GridScore._leaves(f)]
        if isinstance(sf, _ScaledScoringFunction):
            return [
                leaf
                for side in (sf.left, sf.right)
                if isinstance(side, ScoringFunction)
                for leaf in GridScore._leaves(side)
            ]
        if isinstance(sf, Clamp):
            return GridScore._leaves(sf.scoring_function)
        return [sf]

    def get_dependencies(self) -> list[Dependency]:
        # only the positions: the grid replaces the need for a live KNN query at score time
        return [self._position_dep]

    def _interpolate(self, positions: np.ndarray) -> np.ndarray:
        """The grid score of ``(n_poses, n_atoms, 3)`` positions of the probe's scored atoms."""
        return _trilinear_sum(
            np.ascontiguousarray(positions, dtype=np.float64),
            self._values,
            self._columns,
            self._origin,
            self._step,
        )

    def _score(self, pose, computed) -> float:
        positions = computed[self._position_dep][self._probe_mask]
        return float(self._interpolate(positions[None])[0])

    def _batch_scores(self, poses, computed_batch) -> np.ndarray:
        return self._interpolate(computed_batch[self._position_dep][:, self._probe_mask])
