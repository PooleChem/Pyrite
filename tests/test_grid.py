"""`GridScore`: the trilinear lookup equals SciPy's interpolator, also outside and on the edges."""

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from pyrite.scoring import Gaussian, Hydrophobic, Repulsion
from pyrite.scoring.grid import GridScore


@pytest.fixture(scope="module")
def grid(ctx) -> GridScore:
    exact = Gaussian(ctx.ligand, ctx.receptor, k=100) + 0.5 * Repulsion(
        ctx.ligand, ctx.receptor, k=100
    )
    exact = exact + Hydrophobic(ctx.ligand, ctx.receptor, k=100)
    return GridScore(exact, ctx.site, spacing=1.0)


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
