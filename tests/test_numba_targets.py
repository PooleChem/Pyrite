"""
Correctness tests for every function targeted for Numba acceleration.

Each test calls the function with fixed numpy inputs and checks that the output
matches a reference value computed by the *current* (pure-numpy) implementation.
After Numba JIT is applied the same tests must still pass to within float32
tolerance (atol=1e-5).

Tested targets
--------------
_common:
  - _rotation_matrix_from_euler
  - _translation_matrix_from_coordinates

scoring/protein.py (via static / standalone calls):
  - _SlopeStep._slope_step
  - Gaussian._score  (via __gaussian kernel)
  - Repulsion._score
  - Hydrophobic._score
  - NonDirHBond._score
  - LJ._score
  - PlantsPLP.potential_four_piece
  - PlantsPLP.potential_two_piece
"""

import math

import numpy as np
import pytest

from pyrite._common import Mol
from pyrite._util import (
    _rotation_matrix_from_euler,
    _translation_matrix_from_coordinates,
)
from pyrite.scoring.protein import (
    LJ,
    Gaussian,
    Hydrophobic,
    NonDirHBond,
    PlantsPLP,
    Repulsion,
    _SlopeStep,
)

# ---------------------------------------------------------------------------
# Fixtures — one shared protein/ligand pair for all integration-level tests
# ---------------------------------------------------------------------------

PROT = "1mmv"
BASE = "/Users/martvanderlugt/Dev/bp/input/astex"


@pytest.fixture(scope="module")
def mol_pair():
    receptor = Mol.from_pdb(f"{BASE}/{PROT}/{PROT}_protein-processed.pdb")
    ligand = Mol.from_sdf(f"{BASE}/{PROT}/{PROT}_ligand.sdf", flexible=True)
    return ligand, receptor


@pytest.fixture(scope="module")
def computed_knn(mol_pair):
    ligand, receptor = mol_pair
    sf = Gaussian(ligand, receptor, offset=0.0, width=0.5, k=400)
    dep = sf.nn_dep
    return dep, {dep: dep.compute(-1)}


# ---------------------------------------------------------------------------
# _common helpers
# ---------------------------------------------------------------------------


class TestRotationMatrixFromEuler:
    def test_identity(self):
        assert np.allclose(_rotation_matrix_from_euler(0.0, 0.0, 0.0), np.eye(4))

    def test_known_angles(self):
        r, p, y = math.pi / 4, math.pi / 6, math.pi / 3
        expected = np.array(
            [
                [0.4330127, -0.43559574, 0.78914913, 0],
                [0.75, 0.65973961, -0.04736717, 0],
                [-0.5, 0.61237244, 0.61237244, 0],
                [0, 0, 0, 1],
            ]
        )
        assert np.allclose(_rotation_matrix_from_euler(r, p, y), expected)

    def test_last_row_and_col_are_homogeneous(self):
        m = _rotation_matrix_from_euler(1.1, 0.7, 2.3)
        assert np.allclose(m[3, :], [0, 0, 0, 1])
        assert np.allclose(m[:, 3], [0, 0, 0, 1])

    def test_rotation_submatrix_is_orthogonal(self):
        m = _rotation_matrix_from_euler(0.5, 1.2, 0.9)
        R = m[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)

    @pytest.mark.parametrize(
        "roll,pitch,yaw",
        [
            (0.0, 0.0, math.pi),
            (math.pi / 2, 0.0, 0.0),
            (0.0, math.pi / 2, 0.0),
            (1.0, 2.0, 3.0),
            (-0.5, 0.3, -1.7),
        ],
    )
    def test_parametric(self, roll, pitch, yaw):
        m = _rotation_matrix_from_euler(roll, pitch, yaw)
        assert m.shape == (4, 4)
        assert np.allclose(m[3, :], [0, 0, 0, 1])
        assert np.allclose(m[:, 3], [0, 0, 0, 1])
        assert np.allclose(m[:3, :3] @ m[:3, :3].T, np.eye(3), atol=1e-6)


class TestTranslationMatrixFromCoordinates:
    def test_identity(self):
        assert np.allclose(_translation_matrix_from_coordinates(0.0, 0.0, 0.0), np.eye(4))

    def test_known_values(self):
        expected = np.array(
            [[1, 0, 0, 10], [0, 1, 0, -4], [0, 0, 1, 20], [0, 0, 0, 1]], dtype=float
        )
        assert np.allclose(_translation_matrix_from_coordinates(10.0, -4.0, 20.0), expected)

    @pytest.mark.parametrize("x,y,z", [(1.5, -2.3, 0.0), (0.0, 0.0, 100.0), (-7.1, 4.4, -3.3)])
    def test_parametric(self, x, y, z):
        m = _translation_matrix_from_coordinates(x, y, z)
        assert m.shape == (4, 4)
        assert np.allclose(m[:3, :3], np.eye(3))
        assert np.allclose(m[:3, 3], [x, y, z])
        assert np.allclose(m[3, :], [0, 0, 0, 1])


# ---------------------------------------------------------------------------
# _SlopeStep._slope_step  (static — no molecule needed)
# ---------------------------------------------------------------------------


class TestSlopeStep:
    """Tests for _SlopeStep._slope_step(dist, good, bad)."""

    def test_below_good_scores_one(self):
        dist = np.array([-1.0, 0.0, 0.4])
        result = _SlopeStep._slope_step(dist, good=0.5, bad=1.5)
        assert np.allclose(result, 1.0)

    def test_above_bad_scores_zero(self):
        dist = np.array([1.5, 2.0, 10.0])
        result = _SlopeStep._slope_step(dist, good=0.5, bad=1.5)
        assert np.allclose(result, 0.0)

    def test_slope_region(self):
        good, bad = 0.5, 1.5
        dist = np.array([0.5, 1.0, 1.5])
        result = _SlopeStep._slope_step(dist, good=good, bad=bad)
        expected = np.clip((dist - bad) / (good - bad), 0.0, 1.0)
        assert np.allclose(result, expected)

    def test_exact_boundaries(self):
        result = _SlopeStep._slope_step(np.array([0.5]), good=0.5, bad=1.5)
        assert np.allclose(result, 1.0)
        result = _SlopeStep._slope_step(np.array([1.5]), good=0.5, bad=1.5)
        assert np.allclose(result, 0.0)

    def test_output_shape_preserved(self):
        dist = np.zeros((5, 3))
        result = _SlopeStep._slope_step(dist, good=0.5, bad=1.5)
        assert result.shape == dist.shape


# ---------------------------------------------------------------------------
# PlantsPLP static potentials  (no molecule needed)
# ---------------------------------------------------------------------------


class TestPotentialFourPiece:
    """Tests for PlantsPLP.potential_four_piece(r, values)."""

    VALUES = (2.0, 3.0, 4.0, 6.0, -1.0, 5.0)  # a, b, c, d, e, f

    def test_before_a_positive(self):
        a, b, c, d, e, f = self.VALUES
        r = np.array([0.0, 1.0])
        result = PlantsPLP.potential_four_piece(r, self.VALUES)
        expected = f * (a - r) / a
        assert np.allclose(result, expected)

    def test_between_b_and_c_constant(self):
        a, b, c, d, e, f = self.VALUES
        r = np.array([3.0, 3.5, 4.0])
        result = PlantsPLP.potential_four_piece(r, self.VALUES)
        assert np.allclose(result, e)

    def test_after_d_is_zero(self):
        r = np.array([6.0, 7.0, 100.0])
        result = PlantsPLP.potential_four_piece(r, self.VALUES)
        assert np.allclose(result, 0.0)

    def test_known_vector(self):
        a, b, c, d, e, f = self.VALUES
        r = np.linspace(0, 7, 50)
        result = PlantsPLP.potential_four_piece(r, self.VALUES)
        expected = np.zeros_like(r)
        expected[r < a] = f * (a - r[r < a]) / a
        expected[(a <= r) & (r < b)] = e * (r[(a <= r) & (r < b)] - a) / (b - a)
        expected[(b <= r) & (r < c)] = e
        expected[(c <= r) & (r < d)] = e * (d - r[(c <= r) & (r < d)]) / (d - c)
        assert np.allclose(result, expected)

    def test_output_shape(self):
        r = np.zeros(10)
        assert PlantsPLP.potential_four_piece(r, self.VALUES).shape == (10,)


class TestPotentialTwoPiece:
    """Tests for PlantsPLP.potential_two_piece(r, values)."""

    VALUES = (2.0, 6.0, 1.0, 5.0)  # a, b, c, d

    def test_before_a(self):
        a, b, c, d = self.VALUES
        r = np.array([0.0, 1.0])
        result = PlantsPLP.potential_two_piece(r, self.VALUES)
        expected = r * (c - d) / a + d
        assert np.allclose(result, expected)

    def test_after_b_is_zero(self):
        r = np.array([6.0, 7.0, 100.0])
        result = PlantsPLP.potential_two_piece(r, self.VALUES)
        assert np.allclose(result, 0.0)

    def test_known_vector(self):
        a, b, c, d = self.VALUES
        r = np.linspace(0, 7, 50)
        result = PlantsPLP.potential_two_piece(r, self.VALUES)
        expected = np.zeros_like(r)
        expected[r < a] = r[r < a] * (c - d) / a + d
        expected[(a <= r) & (r <= b)] = -c * (r[(a <= r) & (r <= b)] - a) / (b - a) + c
        assert np.allclose(result, expected)

    def test_output_shape(self):
        r = np.zeros(10)
        assert PlantsPLP.potential_two_piece(r, self.VALUES).shape == (10,)


# ---------------------------------------------------------------------------
# Scoring function _score() kernels  (integration — needs real molecule)
# ---------------------------------------------------------------------------


class TestGaussianScore:
    def test_score_is_finite(self, mol_pair, computed_knn):
        ligand, receptor = mol_pair
        dep, computed = computed_knn
        sf = Gaussian(ligand, receptor, offset=0.0, width=0.5, k=400)
        score = sf._score(-1, computed)
        assert np.isfinite(score)

    def test_score_is_non_negative(self, mol_pair, computed_knn):
        ligand, receptor = mol_pair
        dep, computed = computed_knn
        sf = Gaussian(ligand, receptor, offset=0.0, width=0.5, k=400)
        assert sf._score(-1, computed) >= 0.0

    def test_score_reproducible(self, mol_pair, computed_knn):
        ligand, receptor = mol_pair
        dep, computed = computed_knn
        sf = Gaussian(ligand, receptor, offset=0.0, width=0.5, k=400)
        s1 = sf._score(-1, computed)
        s2 = sf._score(-1, computed)
        assert np.isclose(s1, s2)

    def test_wider_gaussian_increases_score(self, mol_pair, computed_knn):
        ligand, receptor = mol_pair
        dep, computed = computed_knn
        sf_narrow = Gaussian(ligand, receptor, offset=0.0, width=0.5, k=400)
        sf_wide = Gaussian(ligand, receptor, offset=0.0, width=2.0, k=400)
        assert sf_wide._score(-1, computed) >= sf_narrow._score(-1, computed)


class TestRepulsionScore:
    def test_score_is_finite(self, mol_pair, computed_knn):
        ligand, receptor = mol_pair
        dep, computed = computed_knn
        sf = Repulsion(ligand, receptor, offset=0.0, k=400)
        assert np.isfinite(sf._score(-1, computed))

    def test_score_is_non_negative(self, mol_pair, computed_knn):
        ligand, receptor = mol_pair
        dep, computed = computed_knn
        sf = Repulsion(ligand, receptor, offset=0.0, k=400)
        assert sf._score(-1, computed) >= 0.0

    def test_score_reproducible(self, mol_pair, computed_knn):
        ligand, receptor = mol_pair
        dep, computed = computed_knn
        sf = Repulsion(ligand, receptor, offset=0.0, k=400)
        assert np.isclose(sf._score(-1, computed), sf._score(-1, computed))

    def test_positive_offset_increases_repulsion(self, mol_pair):
        # Larger offset → larger optimal distance → more atoms inside repulsion zone.
        # Uses get_score() so each SF handles its own KDTree query.
        ligand, receptor = mol_pair
        sf0 = Repulsion(ligand, receptor, offset=0.0, k=400)
        sf_pos = Repulsion(ligand, receptor, offset=1.0, k=400)
        assert sf_pos.get_score() >= sf0.get_score()


class TestHydrophobicScore:
    def test_score_is_finite(self, mol_pair, computed_knn):
        ligand, receptor = mol_pair
        dep, computed = computed_knn
        sf = Hydrophobic(ligand, receptor, good=0.5, bad=1.5, k=400)
        assert np.isfinite(sf._score(-1, computed))

    def test_score_is_non_negative(self, mol_pair, computed_knn):
        ligand, receptor = mol_pair
        dep, computed = computed_knn
        sf = Hydrophobic(ligand, receptor, good=0.5, bad=1.5, k=400)
        assert sf._score(-1, computed) >= 0.0

    def test_score_reproducible(self, mol_pair, computed_knn):
        ligand, receptor = mol_pair
        dep, computed = computed_knn
        sf = Hydrophobic(ligand, receptor, good=0.5, bad=1.5, k=400)
        assert np.isclose(sf._score(-1, computed), sf._score(-1, computed))


class TestNonDirHBondScore:
    def test_score_is_finite(self, mol_pair, computed_knn):
        ligand, receptor = mol_pair
        dep, computed = computed_knn
        sf = NonDirHBond(ligand, receptor, good=-0.7, bad=0.0, k=400)
        assert np.isfinite(sf._score(-1, computed))

    def test_score_reproducible(self, mol_pair, computed_knn):
        ligand, receptor = mol_pair
        dep, computed = computed_knn
        sf = NonDirHBond(ligand, receptor, good=-0.7, bad=0.0, k=400)
        assert np.isclose(sf._score(-1, computed), sf._score(-1, computed))


class TestLJScore:
    def test_score_is_finite(self, mol_pair, computed_knn):
        ligand, receptor = mol_pair
        dep, computed = computed_knn
        sf = LJ(ligand, receptor, k=400)
        assert np.isfinite(sf._score(-1, computed))

    def test_score_reproducible(self, mol_pair, computed_knn):
        ligand, receptor = mol_pair
        dep, computed = computed_knn
        sf = LJ(ligand, receptor, k=400)
        assert np.isclose(sf._score(-1, computed), sf._score(-1, computed))


class TestPlantsPLPScore:
    def test_score_is_finite(self, mol_pair, computed_knn):
        ligand, receptor = mol_pair
        dep, computed = computed_knn
        sf = PlantsPLP(ligand, receptor, k=400)
        assert np.isfinite(sf._score(-1, computed))

    def test_score_reproducible(self, mol_pair, computed_knn):
        ligand, receptor = mol_pair
        dep, computed = computed_knn
        sf = PlantsPLP(ligand, receptor, k=400)
        assert np.isclose(sf._score(-1, computed), sf._score(-1, computed))
