"""Analytic gradients: the pose gradient kernel, `GridScore` and `InternalOverlap`.

The conformance suite checks every scoring function's gradient against finite differences at the
fixture's settings; these tests add the settings it does not cover: other grid spacings, quaternion
layouts and unnormalised quaternions.
"""

import numpy as np
import pytest
from conftest import EXAMPLES

from pyrite import Mol, Pose
from pyrite.bounds import RectangularBounds
from pyrite.scoring import Gaussian, InternalOverlap, Repulsion
from pyrite.scoring.grid import GridScore


def _finite_differences(function, values, step=1e-6):
    gradient = np.empty(len(values))
    for k in range(len(values)):
        shift = np.zeros(len(values))
        shift[k] = step
        gradient[k] = (function(values + shift) - function(values - shift)) / (2 * step)
    return gradient


def _poses(mol: Mol, n: int, seed: int = 0, unnormalised: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    layout = mol.layout
    values = np.tile(np.asarray(mol.input_pose), (n, 1))
    values[:, layout.trans_slice] += rng.normal(0, 1.0, (n, 3))
    values[:, layout.tors_slice] += rng.normal(0, 0.5, (n, layout.n_tors))
    values[:, layout.rot_slice] = layout.sample_random_rotations(n, rng=rng)
    if unnormalised and layout.rot_type == "quat":
        values[:, layout.rot_slice] *= rng.uniform(0.3, 3.0, (n, 1))
    return values


@pytest.fixture(scope="module")
def receptor() -> Mol:
    return Mol.from_pdb(str(EXAMPLES / "factor_x.pdb"))


def _ligand(rot_type: str) -> Mol:
    return Mol.from_sdf(
        str(EXAMPLES / "factor_x_ligand.sdf"), flexible=True, rotation_type=rot_type
    )


@pytest.mark.parametrize("rot_type", ["euler", "quat"])
def test_the_pose_gradient_of_any_forces_is_their_projection_on_the_motion(rot_type):
    # sum_j F_j . dx_j/dpose, for arbitrary forces, against differentiating the positions
    mol = _ligand(rot_type)
    rng = np.random.default_rng(1)
    for values in _poses(mol, 6, unnormalised=True):
        forces = rng.normal(size=(mol.n_atoms, 3))
        expected = _finite_differences(
            lambda v, f=forces: float(np.sum(f * mol.pose_to_positions(v))), values
        )
        got = mol.pose_gradient(values, mol.pose_to_positions(values), forces)
        assert np.allclose(got, expected, atol=1e-6 * max(1.0, np.abs(expected).max()))


_GRIDS = {}


def _grid(receptor: Mol, rot_type: str, interpolation: str) -> tuple[Mol, GridScore]:
    """One grid per layout and interpolation, at 0.75 A (not 1: the spacing must be divided out)."""
    key = (rot_type, interpolation)
    if key not in _GRIDS:
        ligand = _ligand(rot_type)
        exact = Gaussian(ligand, receptor, offset=0.0, width=0.5) + 0.8 * Repulsion(
            ligand, receptor
        )
        box = RectangularBounds.autobox(ligand, padding=2.0)
        _GRIDS[key] = ligand, GridScore(exact, box, spacing=0.75, interpolation=interpolation)
    return _GRIDS[key]


@pytest.mark.filterwarnings("ignore:from_pdb")
@pytest.mark.parametrize("rot_type", ["euler", "quat"])
@pytest.mark.parametrize("interpolation", ["tricubic", "trilinear"])
def test_the_grid_gradient_equals_finite_differences(receptor, rot_type, interpolation):
    ligand, grid = _grid(receptor, rot_type, interpolation)

    for values in _poses(ligand, 6, unnormalised=True):
        pose = Pose(values, ligand.layout)
        score, gradient = grid.get_score_and_gradient(pose)
        expected = _finite_differences(lambda v: grid.get_score(Pose(v, ligand.layout)), values)
        assert score == pytest.approx(grid.get_score(pose), abs=1e-12)
        assert np.allclose(gradient, expected, atol=1e-5 * max(1.0, np.abs(expected).max()))


@pytest.mark.parametrize("rot_type", ["euler", "quat"])
def test_the_internal_overlap_gradient_equals_finite_differences(rot_type):
    ligand = _ligand(rot_type)
    overlap = InternalOverlap(ligand)
    values = _poses(ligand, 30, seed=2, unnormalised=True)
    values[:, ligand.layout.tors_slice] = np.random.default_rng(2).uniform(
        -np.pi, np.pi, (30, ligand.n_tors)
    )  # random torsions: most poses fold onto themselves

    overlapping = 0
    for v in values:
        score, gradient = overlap.get_score_and_gradient(Pose(v, ligand.layout))
        expected = _finite_differences(lambda w: overlap.get_score(Pose(w, ligand.layout)), v)
        overlapping += score > 0
        assert np.allclose(gradient, expected, atol=1e-5 * max(1.0, np.abs(expected).max()))
    assert overlapping > 10


@pytest.mark.filterwarnings("ignore:from_pdb")
def test_a_grid_search_with_the_analytic_gradient_reaches_the_same_minimum(receptor):
    # The gradient is for L-BFGS-B: with jac=True it must find what finite differences find.
    from scipy.optimize import minimize

    ligand, grid = _grid(receptor, "euler", "tricubic")
    function = grid + 0.5 * InternalOverlap(ligand)
    start = _poses(ligand, 1, seed=4)[0]

    def score(v):
        return function.get_score(Pose(v, ligand.layout))

    def score_and_gradient(v):
        return function.get_score_and_gradient(Pose(v, ligand.layout))

    numeric = minimize(score, start, method="L-BFGS-B")
    analytic = minimize(score_and_gradient, start, method="L-BFGS-B", jac=True)

    assert analytic.fun == pytest.approx(numeric.fun, abs=1e-3)
    assert analytic.nfev < numeric.nfev / 5  # one evaluation per step, not one per variable


@pytest.mark.filterwarnings("ignore:from_pdb")
def test_basin_hopping_takes_the_gradient_through_its_minimizer(receptor):
    # BasinHopping only calls its function inside scipy's minimize, so jac=True is all it needs.
    from pyrite.search import BasinHopping, random_hop

    ligand, grid = _grid(receptor, "euler", "tricubic")
    function = grid + 0.5 * InternalOverlap(ligand)
    box = RectangularBounds.autobox(ligand, padding=1.0)
    hopping = BasinHopping(
        function.get_score_and_gradient,
        random_hop(box.get_translation_bounds()),
        T=1.0,
        stepsize=0.5,
        minimizer_kwargs={"method": "L-BFGS-B", "jac": True},
        rng=np.random.default_rng(0),
    )

    result = hopping.run(ligand.input_pose, niter=3)

    assert isinstance(result.fun, float)
    assert result.fun == pytest.approx(function.get_score(result.x))
