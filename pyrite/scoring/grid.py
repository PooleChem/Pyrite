import numpy as np
from scipy.interpolate import RegularGridInterpolator

from pyrite.bounds import Bounds
from pyrite.scoring import ScoringFunction, Clamp
from pyrite.scoring._base import _CombinedScoringFunction, _ScaledScoringFunction
from pyrite.scoring.dependencies import Dependency, KNNDependency


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

        merged = Dependency.merge_all(scoring_function.get_dependencies())
        knn_dep = next(iter(merged))

        axes, shape, grid_points = self._make_grid_points(binding_site, spacing, padding)

        # same KNNDependency machinery every other scoring function already
        # goes through — just point its query at grid vertices instead of
        # the real ligand. Reuses the already-built/cached KDTree (same
        # point_cloud -> same tree_hash -> KDTreeCache hit).
        grid_dep = KNNDependency(knn_dep.point_cloud, lambda conf_id: grid_points, knn_dep.k, knn_dep.distance_upper_bound)
        r_grid, idx_grid, _ = grid_dep.compute(conf_id=None)

        atom_types = sorted(set(self._probe_mol.atom_types[self._probe_mask].tolist()))
        self._interpolators = {}
        for atom_type in atom_types:
            values = scoring_function._score_field(r_grid, idx_grid, atom_type).reshape(shape)
            self._interpolators[atom_type] = RegularGridInterpolator(
                axes, values, bounds_error=False, fill_value=0.0,
            )

    def _make_grid_points(self, binding_site: Bounds, spacing: float, padding: float):
        # how far this ligand can reach from wherever its center atom ends up,
        # independent of rotation — covers the same gap the prototype's
        # sampled-envelope approach was closing, without needing to sample
        center = self._probe_mol.positions[self._probe_mol.center_atom]
        reach = np.linalg.norm(
            self._probe_mol.positions[self._probe_mask] - center, axis=1
        ).max()

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
        return []  # the grid itself replaces the need for a live KNN query at score time

    def _score(self, conf_id, computed) -> float:
        positions = self._probe_mol.get_positions(conf_id)[self._probe_mask]
        types = self._probe_mol.atom_types[self._probe_mask]

        total = 0.0
        for atom_type, interpolator in self._interpolators.items():
            m = types == atom_type
            if m.any():
                total += interpolator(positions[m]).sum()
        return total

    def _batch_scores(self, conf_ids, computed_batch) -> np.ndarray:
        # positions still needs a per-conf_id RDKit read (no numpy-native pose
        # pipeline yet — see POSE_NATIVE_SCORING_PLAN.md), but the interpolator
        # calls below — the actual thing this batches — run once per atom type
        # across every pose at once, not once per atom type *per pose*.
        positions = np.stack([
            self._probe_mol.get_positions(conf_id)[self._probe_mask]
            for conf_id in conf_ids
        ])  # (n_poses, n_atoms, 3)
        types = self._probe_mol.atom_types[self._probe_mask]  # (n_atoms,) — same every pose

        total = np.zeros(len(conf_ids))
        for atom_type, interpolator in self._interpolators.items():
            m = types == atom_type
            if not m.any():
                continue
            pts = positions[:, m]  # (n_poses, n_atoms_of_type, 3)
            n_poses, n_of_type, _ = pts.shape
            values = interpolator(pts.reshape(-1, 3)).reshape(n_poses, n_of_type)
            total += values.sum(axis=1)
        return total
