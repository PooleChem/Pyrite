"""`GridScore`: the trilinear lookup equals SciPy's interpolator, also outside and on the edges."""

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from pyrite.scoring import Gaussian, Hydrophobic, NonDirHBond, Repulsion
from pyrite.scoring.grid import GridScore


@pytest.fixture(scope="module")
def grid(ctx) -> GridScore:
    exact = Gaussian(ctx.ligand, ctx.receptor, k=100) + 0.5 * Repulsion(
        ctx.ligand, ctx.receptor, k=100
    )
    exact = exact + Hydrophobic(ctx.ligand, ctx.receptor, k=100)
    return GridScore(
        exact, ctx.site, spacing=1.0, interpolation="trilinear"
    )  # the lookups tested below


def _reference(grid: GridScore, positions: np.ndarray) -> np.ndarray:
    """Score ``(n, atoms, 3)`` positions with one SciPy interpolator per atom type (0 outside)."""
    axes = [
        origin + step * np.arange(n)
        for origin, step, n in zip(grid._origin, grid._step, grid._values.shape[:3], strict=True)
    ]
    total = np.zeros(len(positions))
    for column in range(grid._values.shape[-1]):
        interpolator = RegularGridInterpolator(
            axes, grid._values[..., column], bounds_error=False, fill_value=0.0
        )
        atoms = grid._columns == column
        total += interpolator(positions[:, atoms].reshape(-1, 3)).reshape(len(positions), -1).sum(1)
    return total


def test_lookup_equals_scipy_inside_and_outside_the_grid(grid):
    rng = np.random.default_rng(0)
    low = grid._origin - 3.0
    high = grid._origin + grid._step * (np.array(grid._values.shape[:3]) - 1) + 3.0
    n_atoms = len(grid._columns)
    positions = rng.uniform(low, high, (400, n_atoms, 3))  # a margin of 3 A outside on each side

    got = grid._interpolate(positions)

    assert np.allclose(got, _reference(grid, positions), rtol=1e-9, atol=1e-9)
    assert (got != 0).any() and (_reference(grid, positions) == 0).sum() == (got == 0).sum()


def test_lookup_on_vertices_and_edges(grid):
    # Points exactly on a vertex, and exactly on the first and last plane, are inside the grid.
    shape = np.array(grid._values.shape[:3])
    rng = np.random.default_rng(1)
    index = np.stack([rng.integers(0, s, (50, len(grid._columns))) for s in shape], axis=-1)
    positions = grid._origin + index * grid._step
    positions[:10] = grid._origin  # the very first vertex
    positions[10:20] = grid._origin + (shape - 1) * grid._step  # the very last one

    assert np.allclose(grid._interpolate(positions), _reference(grid, positions), atol=1e-9)


def test_a_vertex_returns_the_grid_value_of_its_atom_type(grid):
    # All atoms on one vertex: the score is the sum of every atom's own type's value there.
    vertex = (3, 4, 5)
    position = grid._origin + np.array(vertex) * grid._step
    positions = np.broadcast_to(position, (1, len(grid._columns), 3))

    expected = grid._values[(*vertex, slice(None))][grid._columns].sum()

    assert grid._interpolate(positions)[0] == pytest.approx(expected, rel=1e-12)


def test_poses_far_outside_the_grid_score_zero(ctx, grid):
    far = np.array(ctx.poses)
    far[:, 3:6] += 500.0

    assert np.all(grid.batch_scores(type(ctx.poses)(far, ctx.poses.layout)) == 0.0)


# ---------------------------------------------------------------------------
# Tricubic interpolation
# ---------------------------------------------------------------------------


def _random_grid(shape=(12, 14, 11), columns=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(*shape, columns)), np.array([1.0, -2.0, 3.0]), np.full(3, 0.5)


def test_the_tricubic_kernel_equals_scipys_cubic_b_spline():
    from scipy import ndimage

    from pyrite.scoring.grid import _spline_coefficients, _tricubic_sum

    values, origin, step = _random_grid()
    shape, columns = np.array(values.shape[:3]), np.array([0, 1, 2, 1, 0])
    rng = np.random.default_rng(1)
    points = (
        origin + rng.uniform(0, 1, (300, 5, 3)) * (shape - 1) * step
    )  # in every cell, up to the edges

    got = _tricubic_sum(points, _spline_coefficients(values), columns, origin, step, shape)

    expected = np.zeros(len(points))
    for atom, column in enumerate(columns):
        coordinates = ((points[:, atom] - origin) / step).T
        expected += ndimage.map_coordinates(
            values[..., column], coordinates, order=3, mode="mirror"
        )
    assert np.allclose(got, expected, atol=1e-10)


def test_the_tricubic_kernel_is_exact_at_the_vertices_and_zero_outside():
    from pyrite.scoring.grid import _spline_coefficients, _tricubic_sum

    values, origin, step = _random_grid()
    shape, columns = np.array(values.shape[:3]), np.array([0, 1, 2])
    coefficients = _spline_coefficients(values)
    rng = np.random.default_rng(2)
    index = np.stack([rng.integers(0, s, (40, 3)) for s in shape], axis=-1)  # (40, 3 atoms, 3 axes)

    got = _tricubic_sum(origin + index * step, coefficients, columns, origin, step, shape)

    expected = [sum(values[tuple(index[i, a])][columns[a]] for a in range(3)) for i in range(40)]
    assert np.allclose(got, expected, atol=1e-10)
    outside = np.array([[[100.0, 0.0, 0.0], [origin[0] - 0.01, 0.0, 3.0], [1.0, -2.0, 3.0]]])
    # the third atom is on the first vertex: only it contributes
    assert np.allclose(
        _tricubic_sum(outside, coefficients, columns, origin, step, shape), values[0, 0, 0, 2]
    )


def test_the_tricubic_error_falls_with_the_fourth_power_of_the_spacing_the_trilinear_with_the_second():
    from pyrite.scoring.grid import _spline_coefficients, _tricubic_sum, _trilinear_sum

    def field(x, y, z):
        bump = np.exp(-((x - 3) ** 2 + (y - 3) ** 2 + (z - 3) ** 2) / 2.0)
        return bump + 0.3 * np.sin(1.3 * x) * np.cos(0.9 * y) * np.sin(0.7 * z)

    points = np.random.default_rng(1).uniform(1.5, 4.5, (1500, 1, 3))  # far from the edges
    exact = field(points[:, 0, 0], points[:, 0, 1], points[:, 0, 2])
    errors = {"cubic": [], "linear": []}
    for h in (0.4, 0.2):
        axis = np.arange(0, 6 + h / 2, h)
        values = field(*np.meshgrid(axis, axis, axis, indexing="ij"))[..., None]
        origin, step, shape = np.zeros(3), np.full(3, h), np.full(3, len(axis))
        cubic = _tricubic_sum(
            points, _spline_coefficients(values), np.array([0]), origin, step, shape
        )
        linear = _trilinear_sum(points, values, np.array([0]), origin, step)
        errors["cubic"].append(np.abs(cubic - exact).max())
        errors["linear"].append(np.abs(linear - exact).max())

    assert errors["cubic"][0] / errors["cubic"][1] > 12  # 16 for a fourth power
    assert 3 < errors["linear"][0] / errors["linear"][1] < 5  # 4 for a second power
    assert errors["cubic"][1] < 0.01 * errors["linear"][1]


def test_an_unknown_interpolation_is_rejected(ctx):
    with pytest.raises(ValueError, match="interpolation"):
        GridScore(Gaussian(ctx.ligand, ctx.receptor), ctx.site, interpolation="nearest")


@pytest.mark.parametrize("interpolation", ["tricubic", "trilinear"])
def test_a_grid_score_satisfies_the_scoring_function_contract(ctx, interpolation):
    from contracts import check_scoring_function

    grid = GridScore(
        Gaussian(ctx.ligand, ctx.receptor) + Repulsion(ctx.ligand, ctx.receptor),
        ctx.site,
        spacing=1.0,
        interpolation=interpolation,
    )

    check_scoring_function(grid, ctx.ligand, ctx.poses)


def test_the_default_is_tricubic(ctx):
    # Trilinear scores the wells, where good poses sit, systematically too high, so a search on it
    # finds worse poses: tricubic is the default.
    default = GridScore(Gaussian(ctx.ligand, ctx.receptor), ctx.site, spacing=1.0)
    explicit = GridScore(
        Gaussian(ctx.ligand, ctx.receptor), ctx.site, spacing=1.0, interpolation="tricubic"
    )

    assert default.interpolation == "tricubic"
    assert np.array_equal(default.batch_scores(ctx.poses), explicit.batch_scores(ctx.poses))
    assert default._coefficients is not None


def test_tricubic_removes_the_bias_of_trilinear_where_good_poses_are(ctx):
    # Trilinear interpolation overestimates the wells of the potentials, where good poses sit:
    # near the crystal pose it scores them +6.7 too high on average at spacing 1.0, tricubic -0.04.
    from helpers import loaded_pose

    ligand, receptor, k = ctx.ligand, ctx.receptor, 100
    exact = (
        -0.035579 * Gaussian(ligand, receptor, offset=0.0, width=0.5, k=k)
        + -0.005156 * Gaussian(ligand, receptor, offset=3.0, width=2.0, k=k)
        + 0.840245 * Repulsion(ligand, receptor, offset=0.0, k=k)
        + -0.035069 * Hydrophobic(ligand, receptor, good=0.5, bad=1.5, k=k)
        + -0.587439 * NonDirHBond(ligand, receptor, good=-0.7, bad=0.0, k=k)
    )
    layout, rng = ligand.layout, np.random.default_rng(0)
    noise = np.zeros((100, layout.n_dims))
    noise[:, layout.trans_slice] = rng.normal(0, 0.4, (100, 3))
    noise[:, layout.tors_slice] = rng.normal(0, 0.25, (100, layout.n_tors))
    poses = type(ctx.poses)(np.array(loaded_pose(ligand)) + noise, layout)
    expected = exact.batch_scores(poses)

    errors = {
        kind: GridScore(exact, ctx.site, spacing=1.0, interpolation=kind).batch_scores(poses)
        - expected
        for kind in ("trilinear", "tricubic")
    }

    assert errors["trilinear"].mean() > 3.0  # systematically too high
    assert abs(errors["tricubic"].mean()) < 0.5
    assert np.abs(errors["tricubic"]).mean() < 0.2 * np.abs(errors["trilinear"]).mean()
