"""`Mol.input_pose` and `Mol.pose_from_positions`: poses of given coordinates."""

import numpy as np
import pytest
from conftest import EXAMPLES
from rdkit import Chem
from rdkit.Chem import AllChem

from pyrite import Mol, Poses

LIGAND = str(EXAMPLES / "factor_x_ligand.sdf")


@pytest.fixture(params=["euler", "quat"])
def mol(request) -> Mol:
    return Mol.from_sdf(LIGAND, flexible=True, hydrogens="keep", rotation_type=request.param)


def _random_poses(mol: Mol, n: int = 10) -> Poses:
    rng = np.random.default_rng(0)
    rotations = mol.layout.sample_random_rotations(n, rng=rng)
    translations = rng.uniform(-5, 5, (n, 3))
    torsions = mol.layout.sample_random_torsions(n, rng=rng)
    return Poses.from_parts(rotations, translations, torsions, layout=mol.layout)


def test_the_input_pose_puts_every_atom_where_the_input_had_it(mol):
    expected = Chem.MolFromMolFile(LIGAND, removeHs=False, sanitize=False).GetConformer()

    assert np.allclose(mol.pose_to_positions(mol.input_pose), expected.GetPositions(), atol=1e-9)


def test_the_input_pose_does_not_follow_the_global_conformer(mol):
    before = np.array(mol.input_pose)
    moved = _random_poses(mol, 1)[0]
    mol.pose_to_conformer(moved)  # moves the global conformer (export / display)

    assert np.allclose(mol.get_positions(), mol.pose_to_positions(moved))  # it did move
    assert np.array_equal(np.array(mol.input_pose), before)


def test_the_input_pose_follows_a_new_center_atom(mol):
    input_positions = mol.pose_to_positions(mol.input_pose)
    mol.center_atom = (mol.center_atom + 3) % mol.n_atoms

    assert np.allclose(mol.pose_to_positions(mol.input_pose), input_positions, atol=1e-9)


def test_a_rigid_molecule_has_an_input_pose_too():
    rigid = Mol.from_sdf(LIGAND, flexible=False)

    assert rigid.input_pose.torsions.size == 0
    assert np.allclose(rigid.pose_to_positions(rigid.input_pose), rigid.get_positions())


def test_pose_from_positions_recovers_any_pose(mol):
    for pose in _random_poses(mol):
        positions = mol.pose_to_positions(pose)

        recovered = mol.pose_from_positions(positions)

        assert np.allclose(mol.pose_to_positions(recovered), positions, atol=1e-8)


def test_pose_from_positions_accepts_a_slightly_different_geometry_and_refuses_another():
    # Bond lengths and angles are not part of a pose: an MMFF-relaxed copy is close, not equal.
    rd = Chem.MolFromMolFile(LIGAND, removeHs=False)
    AllChem.MMFFOptimizeMolecule(rd, maxIters=5)
    relaxed = rd.GetConformer().GetPositions()
    mol = Mol.from_sdf(LIGAND, flexible=True, hydrogens="keep")

    pose = mol.pose_from_positions(relaxed, rmsd_delta=0.5)

    rmsd = np.sqrt(np.mean(np.sum((mol.pose_to_positions(pose) - relaxed) ** 2, axis=1)))
    assert 0.0 < rmsd < 0.5
    with pytest.raises(ValueError, match="rmsd_delta"):
        mol.pose_from_positions(relaxed, rmsd_delta=rmsd / 2)


@pytest.mark.parametrize("bad", ["shape", "nan"])
def test_pose_from_positions_rejects_unusable_input(mol, bad):
    positions = mol.get_positions()
    positions = (
        positions[:-1]
        if bad == "shape"
        else np.where(np.arange(len(positions))[:, None] == 0, np.nan, positions)
    )

    with pytest.raises(ValueError, match="shape|finite"):
        mol.pose_from_positions(positions)


def test_pose_from_positions_never_returns_a_mirror_image():
    # For flat molecules the best rigid fit can come out as a reflection; it must be a rotation.
    rd = Chem.MolFromSmiles("Oc1ccc(Cl)c(F)c1")  # planar, no hydrogens out of the plane
    AllChem.Compute2DCoords(rd)
    flat = Mol(rd, flexible=False)

    for pose in _random_poses(flat, 30):
        positions = flat.pose_to_positions(pose)
        recovered = flat.pose_from_positions(positions)
        assert np.allclose(flat.pose_to_positions(recovered), positions, atol=1e-8)
