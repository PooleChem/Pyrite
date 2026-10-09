import numpy as np
from numba import njit
from scipy import ndimage

from pyrite.bounds import Bounds
from pyrite.scoring import Clamp, ScoringFunction
from pyrite.scoring._base import _CombinedScoringFunction, _ScaledScoringFunction
from pyrite.scoring.dependencies import Dependency, KNNDependency, PositionDependency

# The number of grid vertices scored at once while building a grid.
_GRID_CHUNK = 1024


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


@njit
def _tricubic_sum(positions, coefficients, columns, origin, step, n_vertices):
    """Sum, per pose, the cubic B-spline interpolation of every atom in its own column.

    The interpolation is exact at the grid vertices, twice continuously differentiable, and its error
    falls with the fourth power of the spacing (that of trilinear interpolation: the second).

    Parameters
    ----------
    positions : ndarray
        Shape ``(n_poses, n_atoms, 3)``.
    coefficients : ndarray
        The B-spline coefficients of the grid values, shape ``(nx + 4, ny + 4, nz + 4, n_columns)``:
        see `_spline_coefficients`, which also pads them.
    columns : ndarray
        Shape ``(n_atoms,)``, the column every atom reads.
    origin, step : ndarray
        Shape ``(3,)``: the first grid vertex, and the (uniform) spacing, per axis.
    n_vertices : ndarray
        Shape ``(3,)``: the number of grid vertices per axis. Atoms outside the grid contribute 0.

    Returns
    -------
    ndarray
        Shape ``(n_poses,)``.
    """
    n_poses, n_atoms, _ = positions.shape
    out = np.zeros(n_poses)
    weights = np.empty((3, 4))
    index = np.empty(3, dtype=np.int64)
    for i in range(n_poses):
        total = 0.0
        for j in range(n_atoms):
            inside = True
            for d in range(3):
                f = (positions[i, j, d] - origin[d]) / step[d]
                if f < 0 or f > n_vertices[d] - 1:
                    inside = False
                    break
                cell = min(int(f), n_vertices[d] - 2)
                t = f - cell
                index[d] = cell + 1  # the first of the four coefficients, in the padded array
                weights[d, 0] = (1 - t) ** 3 / 6.0
                weights[d, 1] = (3 * t**3 - 6 * t**2 + 4) / 6.0
                weights[d, 2] = (-3 * t**3 + 3 * t**2 + 3 * t + 1) / 6.0
                weights[d, 3] = t**3 / 6.0
            if not inside:
                continue
            c = columns[j]
            value = 0.0
            for a in range(4):
                for b in range(4):
                    wab = weights[0, a] * weights[1, b]
                    for k in range(4):
                        value += (
                            wab
                            * weights[2, k]
                            * coefficients[index[0] + a, index[1] + b, index[2] + k, c]
                        )
            total += value
        out[i] = total
    return out


@njit(cache=True)
def _tricubic_forces(positions, coefficients, columns, origin, step, n_vertices):
    """The tricubic score of one pose and its gradient per atom, ``dS/dx``.

    As `_tricubic_sum` for ``(n_atoms, 3)`` positions, with the derivative of the cubic B-spline
    weights along each axis. Atoms outside the grid contribute 0 to both.

    Returns
    -------
    value : float
    forces : ndarray
        Shape ``(n_atoms, 3)``.
    """
    n_atoms = positions.shape[0]
    forces = np.zeros((n_atoms, 3))
    weights = np.empty((3, 4))
    slopes = np.empty((3, 4))
    index = np.empty(3, dtype=np.int64)
    total = 0.0
    for j in range(n_atoms):
        inside = True
        for d in range(3):
            f = (positions[j, d] - origin[d]) / step[d]
            if f < 0 or f > n_vertices[d] - 1:
                inside = False
                break
            cell = min(int(f), n_vertices[d] - 2)
            t = f - cell
            index[d] = cell + 1
            weights[d, 0] = (1 - t) ** 3 / 6.0
            weights[d, 1] = (3 * t**3 - 6 * t**2 + 4) / 6.0
            weights[d, 2] = (-3 * t**3 + 3 * t**2 + 3 * t + 1) / 6.0
            weights[d, 3] = t**3 / 6.0
            # d weight / d position: the derivative in t, divided by the spacing
            slopes[d, 0] = -((1 - t) ** 2) / 2.0 / step[d]
            slopes[d, 1] = (3 * t**2 - 4 * t) / 2.0 / step[d]
            slopes[d, 2] = (-3 * t**2 + 2 * t + 1) / 2.0 / step[d]
            slopes[d, 3] = t**2 / 2.0 / step[d]
        if not inside:
            continue
        c = columns[j]
        value = gx = gy = gz = 0.0
        for a in range(4):
            for b in range(4):
                for k in range(4):
                    coefficient = coefficients[index[0] + a, index[1] + b, index[2] + k, c]
                    value += weights[0, a] * weights[1, b] * weights[2, k] * coefficient
                    gx += slopes[0, a] * weights[1, b] * weights[2, k] * coefficient
                    gy += weights[0, a] * slopes[1, b] * weights[2, k] * coefficient
                    gz += weights[0, a] * weights[1, b] * slopes[2, k] * coefficient
        total += value
        forces[j, 0], forces[j, 1], forces[j, 2] = gx, gy, gz
    return total, forces


@njit(cache=True)
def _trilinear_forces(positions, values, columns, origin, step):
    """The trilinear score of one pose and its gradient per atom (piecewise constant).

    As `_trilinear_sum` for ``(n_atoms, 3)`` positions. Atoms outside the grid contribute 0.
    """
    n_atoms = positions.shape[0]
    nx, ny, nz = values.shape[0], values.shape[1], values.shape[2]
    forces = np.zeros((n_atoms, 3))
    total = 0.0
    for j in range(n_atoms):
        f0 = (positions[j, 0] - origin[0]) / step[0]
        f1 = (positions[j, 1] - origin[1]) / step[1]
        f2 = (positions[j, 2] - origin[2]) / step[2]
        if f0 < 0 or f1 < 0 or f2 < 0 or f0 > nx - 1 or f1 > ny - 1 or f2 > nz - 1:
            continue
        i0, i1, i2 = min(int(f0), nx - 2), min(int(f1), ny - 2), min(int(f2), nz - 2)
        d0, d1, d2 = f0 - i0, f1 - i1, f2 - i2
        c = columns[j]
        v000, v100 = values[i0, i1, i2, c], values[i0 + 1, i1, i2, c]
        v010, v001 = values[i0, i1 + 1, i2, c], values[i0, i1, i2 + 1, c]
        v110, v101 = values[i0 + 1, i1 + 1, i2, c], values[i0 + 1, i1, i2 + 1, c]
        v011, v111 = values[i0, i1 + 1, i2 + 1, c], values[i0 + 1, i1 + 1, i2 + 1, c]
        total += (
            v000 * (1 - d0) * (1 - d1) * (1 - d2)
            + v100 * d0 * (1 - d1) * (1 - d2)
            + v010 * (1 - d0) * d1 * (1 - d2)
            + v001 * (1 - d0) * (1 - d1) * d2
            + v110 * d0 * d1 * (1 - d2)
            + v101 * d0 * (1 - d1) * d2
            + v011 * (1 - d0) * d1 * d2
            + v111 * d0 * d1 * d2
        )
        forces[j, 0] = (
            (v100 - v000) * (1 - d1) * (1 - d2)
            + (v110 - v010) * d1 * (1 - d2)
            + (v101 - v001) * (1 - d1) * d2
            + (v111 - v011) * d1 * d2
        ) / step[0]
        forces[j, 1] = (
            (v010 - v000) * (1 - d0) * (1 - d2)
            + (v110 - v100) * d0 * (1 - d2)
            + (v011 - v001) * (1 - d0) * d2
            + (v111 - v101) * d0 * d2
        ) / step[1]
        forces[j, 2] = (
            (v001 - v000) * (1 - d0) * (1 - d1)
            + (v101 - v100) * d0 * (1 - d1)
            + (v011 - v010) * (1 - d0) * d1
            + (v111 - v110) * d0 * d1
        ) / step[2]
    return total, forces


def _spline_coefficients(values: np.ndarray) -> np.ndarray:
    """The cubic B-spline coefficients of grid values ``(nx, ny, nz, n_columns)``, padded by 2.

    The values outside the grid are taken as the mirror image of the values inside, which only
    matters within a few vertices of the edge (the influence of a vertex decays by a factor 3.7
    per vertex), and atoms outside the grid score 0 anyway.
    """
    coefficients = np.stack(
        [
            ndimage.spline_filter(values[..., c], order=3, mode="mirror")
            for c in range(values.shape[-1])
        ],
        axis=-1,
    )
    padded = np.pad(coefficients, ((2, 2), (2, 2), (2, 2), (0, 0)), mode="reflect")
    return np.ascontiguousarray(padded)


class GridScore(ScoringFunction):
    """
    ✈️ — Approximates a KNN-based scoring function with a precomputed 3D grid.

    The score of every atom type is computed once, on the vertices of a regular grid, and is
    interpolated during a search: no neighbor search and no kernels at score time. A score costs
    about as much as computing the atom positions, and its gradient is analytic, see
    :meth:`~pyrite.scoring.ScoringFunction.get_score_and_gradient`.

    Only scoring functions built from KNN-based terms can be put on a grid: every term must
    depend on the position of one ligand atom relative to the fixed receptor. Terms that depend on
    the ligand's own conformation, like :class:`~pyrite.scoring.InternalOverlap` or :class:`~pyrite.scoring.NumTors`, are combined
    afterwards: ``GridScore(knn_terms, site) + 0.5 * InternalOverlap(ligand)``.

    Below, a cut through what a grid stores: the score of one ligand atom moving towards a
    surface of random receptor atoms (at protein density), summed over all of them, on a grid
    with a spacing of 1.0 A. Trilinear interpolation scores the well, where good poses sit, too
    high everywhere between the vertices; tricubic follows the exact score closely.

    .. plot::
       :width: 80%
       :alt: Grid interpolation example

       import numpy as np
       import matplotlib.pyplot as plt
       from scipy import ndimage

       rng = np.random.default_rng(1)
       atoms = rng.uniform([0, -12, -12], [12, 12, 12], (345, 3))  # 0.05 atoms per A^3


       def score(x):
           r = np.linalg.norm(atoms[None] - np.stack([x, 0 * x, 0 * x], 1)[:, None], axis=2)
           d = r - 3.6
           pair = -0.0356 * np.exp(-((d / 0.5) ** 2)) - 0.0052 * np.exp(-(((d - 3) / 2) ** 2))
           return (pair + 0.84 * np.minimum(d, 0.0) ** 2).sum(axis=1)


       vertices = np.arange(-6.0, -0.5, 1.0)
       values = score(vertices)
       x = np.linspace(-6.0, -1.5, 500)
       coefficients = ndimage.spline_filter1d(values, order=3, mode="mirror")
       tricubic = ndimage.map_coordinates(
           coefficients, [x - vertices[0]], order=3, mode="mirror", prefilter=False
       )

       plt.figure()
       plt.grid(visible=True)
       plt.plot(x, score(x), linewidth=2, color="black", label="exact")
       plt.plot(x, np.interp(x, vertices, values), linewidth=2, linestyle="--", label="trilinear")
       plt.plot(x, tricubic, linewidth=2, label="tricubic")
       plt.plot(vertices[:-1], values[:-1], "o", color="black", label="grid vertices")
       plt.ylim(-0.3, 0.4)
       plt.xlabel("Position (Angstrom)")
       plt.ylabel("Score")
       plt.legend()

    .. note::
        Only terms between ligand and receptor whose score depends on nothing but the type and
        position of every ligand atom can be put on a grid: the terms with a ``_score_field``,
        today :class:`~pyrite.scoring.Gaussian`, :class:`~pyrite.scoring.Repulsion`,
        :class:`~pyrite.scoring.Hydrophobic`, :class:`~pyrite.scoring.NonHydrophobic` and
        :class:`~pyrite.scoring.NonDirHBond`, and sums and multiples of them. The Lennard-Jones,
        charge and PLANTS terms cannot (yet). Terms of the ligand alone, such as
        :class:`~pyrite.scoring.InternalOverlap`, are combined with the grid instead.

    .. note::
        The grid covers the translation bounds of `binding_site`, padded by the largest distance
        of any ligand atom to its center atom (so every rotation fits), plus `padding`. An atom
        outside the grid scores 0, with a gradient of 0.

    .. warning::
        Building the grid is the expensive part: the scoring function is evaluated on every
        vertex, and it all has to fit in memory. The number of vertices grows with the cube of
        ``1 / spacing``: halving the spacing makes the grid 8 times as expensive to build, and a
        larger ``k`` makes every vertex more expensive.

    **Speed**: ✈️ | batched: 🚀, after building.

    Parameters
    ----------
    scoring_function : ScoringFunction
        The KNN-based scoring function to approximate.
    binding_site : Bounds
        The region the ligand will be searched within. The grid covers this, not wherever the
        ligand happens to be when the ``GridScore`` is made.
    spacing : float, default 0.5
        The distance between the grid vertices, in Angstrom.
    padding : float, default 4.0
        Extra padding around the binding site and the reach of the ligand, in Angstrom.
    interpolation : {'tricubic', 'trilinear'}, default 'tricubic'
        How the grid is interpolated between its vertices. Tricubic (a cubic B-spline) is exact at
        the vertices and its error falls with the fourth power of the spacing, so even 1.0 A is
        close to the exact score. Trilinear is slightly cheaper, but its error falls with the
        square of the spacing and is systematic: the wells of the potentials, where good poses
        sit, score too high, so a search finds worse poses.

    Raises
    ------
    ValueError
        If `interpolation` is unknown.
    TypeError
        If a term of `scoring_function` cannot be put on a grid (it has no ``_score_field``): see
        the note above.

    See Also
    --------
    pyrite.scoring.ScoringFunction.get_score_and_gradient : The score and its (analytic) gradient.
    pyrite.search.BasinHopping : A search that can use the gradient.

    Examples
    --------
    >>> from pyrite import Mol
    >>> ligand = Mol.from_sdf("ligand.sdf", flexible=True)
    >>> receptor = Mol.from_pdb("receptor.pdb")
    >>> from pyrite.bounds import RectangularBounds
    >>> from pyrite.scoring import Gaussian, InternalOverlap, NumTors, Repulsion
    >>> from pyrite.scoring.grid import GridScore
    >>> site = RectangularBounds.autobox(ligand, padding=1.0)
    >>> receptor_terms = Gaussian(ligand, receptor) + 0.84 * Repulsion(ligand, receptor)
    >>> grid = GridScore(receptor_terms, site, spacing=0.5)
    >>> scoring_function = (grid + 0.5 * InternalOverlap(ligand)) / (1 + 0.0585 * NumTors(ligand))
    >>> score, gradient = scoring_function.get_score_and_gradient(ligand.input_pose)
    """

    def __init__(
        self,
        scoring_function: ScoringFunction,
        binding_site: Bounds,
        spacing: float = 0.5,
        padding: float = 4.0,
        interpolation: str = "tricubic",
    ):
        if interpolation not in ("trilinear", "tricubic"):
            raise ValueError(
                f"interpolation must be 'tricubic' or 'trilinear', not {interpolation!r}."
            )
        self.interpolation = interpolation
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

        # One column per atom type present in the ligand; every atom reads the column of its type.
        probe_types = self._probe_mol.atom_types[self._probe_mask]
        atom_types = sorted(set(probe_types.tolist()))

        # The grid is scored a chunk of vertices at a time: the arrays of all neighbors of all
        # vertices at once (atom types x vertices x k) do not fit in the CPU caches, and take
        # gigabytes.
        # The same KNNDependency as for a ligand, pointed at the vertices; it reuses the cached
        # KD-tree of the receptor.
        values = np.empty((len(grid_points), len(atom_types)))
        for start in range(0, len(grid_points), _GRID_CHUNK):
            chunk = grid_points[start : start + _GRID_CHUNK]
            chunk_dep = KNNDependency(
                knn_dep.point_cloud,
                lambda _, chunk=chunk: chunk,
                knn_dep.k,
                knn_dep.distance_upper_bound,
            )
            r, idx, _ = chunk_dep.compute(None)
            # All atom types at once, as a column: the terms broadcast them against the
            # (vertices, k) neighbors, so what depends on the receptor alone (the radii and
            # properties of the neighbors) is looked up once per chunk, not once per type.
            scores = scoring_function._score_field(r, idx, np.array(atom_types)[:, None])
            values[start : start + len(chunk)] = scores.T
        self._values = np.ascontiguousarray(values.reshape(*shape, len(atom_types)))
        self._columns = np.array([atom_types.index(t) for t in probe_types.tolist()])
        self._origin = np.array([axis[0] for axis in axes])
        self._step = np.array([axis[1] - axis[0] for axis in axes])
        self._n_vertices = np.array(self._values.shape[:3])
        self._coefficients = (
            _spline_coefficients(self._values) if interpolation == "tricubic" else None
        )

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
        positions = np.ascontiguousarray(positions, dtype=np.float64)
        if self.interpolation == "tricubic":
            return _tricubic_sum(
                positions,
                self._coefficients,
                self._columns,
                self._origin,
                self._step,
                self._n_vertices,
            )
        return _trilinear_sum(positions, self._values, self._columns, self._origin, self._step)

    def _score(self, pose, computed) -> float:
        positions = computed[self._position_dep][self._probe_mask]
        return float(self._interpolate(positions[None])[0])

    def _batch_scores(self, poses, computed_batch) -> np.ndarray:
        return self._interpolate(computed_batch[self._position_dep][:, self._probe_mask])

    def _score_and_gradient(self, pose, computed):
        positions = computed[self._position_dep]
        scored = np.ascontiguousarray(positions[self._probe_mask], dtype=np.float64)
        if self.interpolation == "tricubic":
            value, scored_forces = _tricubic_forces(
                scored,
                self._coefficients,
                self._columns,
                self._origin,
                self._step,
                self._n_vertices,
            )
        else:
            value, scored_forces = _trilinear_forces(
                scored, self._values, self._columns, self._origin, self._step
            )
        forces = np.zeros_like(positions)
        forces[self._probe_mask] = scored_forces
        return value, self._probe_mol.pose_gradient(pose, positions, forces)
