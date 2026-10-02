"""Placement: `Bounds` positions, `place_in`, and the torsion API of `Mol`."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from pyrite import Mol
from pyrite.bounds import RectangularBounds
from pyrite.search import place_in


def _mol(rotation_type: str = "euler") -> Mol:
    rd = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1-c1ccccc1"))
    AllChem.EmbedMolecule(rd, randomSeed=1)
    return Mol(Chem.RemoveHs(rd), flexible=True, rotation_type=rotation_type)


@pytest.fixture(scope="module")
def site() -> RectangularBounds:
    return RectangularBounds.autobox(_mol(), padding=3.0)


def test_bounds_random_positions(site):
    positions = site.place_random_uniform(50, rng=np.random.default_rng(0))

    assert positions.shape == (50, 3)
    assert all(site.is_within(tuple(p)) for p in positions)
    again = site.place_random_uniform(50, rng=np.random.default_rng(0))
    other = site.place_random_uniform(50, rng=np.random.default_rng(1))
    assert np.array_equal(positions, again) and not np.array_equal(positions, other)


def test_bounds_grid_positions(site):
    grid = site.place_grid(3)

    assert grid.shape == (27, 3)
    assert len({tuple(p) for p in grid}) == 27
    lo = np.array([b[0] for b in site.get_translation_bounds()])
    hi = np.array([b[1] for b in site.get_translation_bounds()])
    assert np.allclose(grid.min(axis=0), lo) and np.allclose(grid.max(axis=0), hi)


@pytest.mark.parametrize("rot_type", ["euler", "quat"])
def test_get_bounds_follows_the_layout(site, rot_type):
    layout = _mol(rot_type).layout
    bounds = site.get_bounds(layout)

    assert len(bounds) == layout.n_dims
    assert bounds[layout.trans_slice] == site.get_translation_bounds()
    rotation_bound = (-1.0, 1.0) if rot_type == "quat" else (-2 * np.pi, 2 * np.pi)
    assert bounds[layout.rot_slice] == [rotation_bound] * layout.rot_dim
    assert bounds[layout.tors_slice] == [(-2 * np.pi, 2 * np.pi)] * layout.n_tors
    assert site.get_bounds(layout, rotation_bounds=(0, 1), torsion_bounds=(-1, 1))[0] == (0, 1)


@pytest.mark.parametrize("rot_type", ["euler", "quat"])
def test_place_in_works_for_every_layout(site, rot_type):
    mol = _mol(rot_type)
    poses = place_in(mol, site, 12, 5, rng=np.random.default_rng(0))

    assert len(poses) == 12 and poses.layout == mol.layout
    assert poses._vs.shape == (12, mol.layout.n_dims)
    assert all(site.is_within(tuple(t)) for t in poses.translation)
    if rot_type == "quat":
        assert np.allclose(np.linalg.norm(poses.rotation, axis=1), 1.0)
    # The translation is the position of the center atom, so a pose really is in the site.
    for pose in list(poses)[:3]:
        conf_id = mol.pose_to_conformer(pose, new_conf=True)
        assert np.allclose(mol.get_positions(conf_id)[mol._center_atom], pose.translation)
        mol.RemoveConformer(conf_id)


@pytest.mark.parametrize(
    ("placement", "conformations", "combine", "n_positions", "n_conformations", "expected"),
    [
        ("random", "conformer", "random", 8, 3, 8),
        ("random", "conformer", "random", 3, 8, 8),
        ("random", "random", "random", 6, 6, 6),
        ("random", "random", "grid", 4, 3, 12),
        ("grid", "random", "grid", 3, 2, 54),
        ("grid", "conformer", "random", 2, 5, 8),
    ],
)
def test_place_in_sizes(
    site, placement, conformations, combine, n_positions, n_conformations, expected
):
    poses = place_in(
        _mol(),
        site,
        n_positions,
        n_conformations,
        placement=placement,
        conformations=conformations,
        combine=combine,
        rng=np.random.default_rng(0),
    )

    assert len(poses) == expected


def test_place_in_grid_gives_every_position_its_own_orientation(site):
    poses = place_in(
        _mol(), site, 2, 1, placement="grid", combine="grid", rng=np.random.default_rng(0)
    )

    assert len({tuple(r) for r in poses.rotation}) == len(poses) == 8


def test_place_in_is_reproducible_from_the_rng(site):
    mol = _mol()
    a = place_in(mol, site, 10, 4, rng=np.random.default_rng(3))
    b = place_in(mol, site, 10, 4, rng=np.random.default_rng(3))
    c = place_in(mol, site, 10, 4, rng=np.random.default_rng(4))

    assert a == b and not a == c
    assert not np.array_equal(a.torsions, c.torsions)  # the conformers follow the rng, too


def test_place_in_validates_its_options(site):
    mol = _mol()
    for bad in ({"placement": "x"}, {"conformations": "x"}, {"combine": "x"}):
        with pytest.raises(ValueError):
            place_in(mol, site, 3, 3, **bad)


def test_torsion_api():
    mol = _mol()

    assert mol.n_tors == 1 == len(mol.rotatable_torsions) == len(mol.torsions)
    mol.set_torsions([0.5])
    assert mol.torsions[0] == pytest.approx(0.5)
    mol.set_torsion(0, -1.0)
    assert mol.torsions[0] == pytest.approx(-1.0)
    for old in ("dihedral_angles", "rotatable_dihedrals", "set_dihedral_angles", "place_in"):
        assert not hasattr(mol, old)
