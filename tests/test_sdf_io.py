"""Writing poses to an SDF file and reading them back: `Mol.to_sdf` and `Mol.poses_from_sdf`."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from pyrite import Mol, PoseLayout, Poses


def _molecule(rotation_type: str = "euler", flexible: bool = True) -> Mol:
    rd = Chem.AddHs(Chem.MolFromSmiles("CCCC(CCO)(CC(=O)OC)c1ccccc1OCC"))
    AllChem.EmbedMolecule(rd, randomSeed=1)
    return Mol(Chem.RemoveHs(rd), flexible=flexible, rotation_type=rotation_type)


def _poses(mol: Mol, n: int, seed: int = 5) -> Poses:
    layout, rng = mol.layout, np.random.default_rng(seed)
    return Poses.from_parts(
        layout,
        layout.sample_random_rotations(n, rng),
        rng.uniform(-5, 5, (n, 3)),
        rng.uniform(-3 * np.pi, 3 * np.pi, (n, layout.n_tors)),  # also beyond +-pi
    )


def _written(path) -> list[np.ndarray]:
    return [m.GetConformer().GetPositions() for m in Chem.SDMolSupplier(str(path), sanitize=False)]


def test_poses_written_to_sdf_are_the_positions_of_the_poses(tmp_path):
    mol, poses = _molecule(), _poses(_molecule(), 4)
    state = mol.rdkit.ToBinary()

    mol.to_sdf(str(tmp_path / "poses.sdf"), poses)

    assert mol.rdkit.ToBinary() == state  # the molecule is left exactly as it was
    assert len(_written(tmp_path / "poses.sdf")) == 4
    assert np.allclose(_written(tmp_path / "poses.sdf"), mol.pose_to_positions(poses), atol=1e-3)


def test_a_single_pose_is_one_record(tmp_path):
    mol = _molecule()
    pose = _poses(mol, 1)[0]

    mol.to_sdf(str(tmp_path / "one.sdf"), pose)

    (record,) = _written(tmp_path / "one.sdf")
    assert np.allclose(record, mol.pose_to_positions(pose), atol=1e-3)


def test_without_poses_the_current_conformer_is_written(tmp_path):
    mol = _molecule()
    mol.to_sdf(str(tmp_path / "now.sdf"))

    (record,) = _written(tmp_path / "now.sdf")
    assert np.allclose(record, mol.get_positions(), atol=1e-3)
    with pytest.raises(ValueError, match="not both"):
        mol.to_sdf(str(tmp_path / "x.sdf"), _poses(mol, 1), conf_id=0)


def test_an_open_writer_is_left_open_and_can_hold_several_calls(tmp_path):
    mol = _molecule()
    writer = Chem.SDWriter(str(tmp_path / "two.sdf"))

    mol.to_sdf(writer, _poses(mol, 2, seed=1))
    mol.to_sdf(writer, _poses(mol, 3, seed=2))
    writer.close()

    assert len(_written(tmp_path / "two.sdf")) == 5


@pytest.mark.parametrize("rotation_type", ["euler", "quat"])
def test_poses_survive_a_round_trip_through_an_sdf_file(rotation_type, tmp_path):
    mol, poses = _molecule(rotation_type), _poses(_molecule(rotation_type), 6)
    path = str(tmp_path / "poses.sdf")
    mol.to_sdf(path, poses)

    loaded = mol.poses_from_sdf(path, rmsd_delta=0.01)

    assert isinstance(loaded, Poses) and loaded.layout == mol.layout and len(loaded) == 6
    # an SDF file has 4 decimals
    assert np.allclose(mol.pose_to_positions(loaded), mol.pose_to_positions(poses), atol=2e-3)
    assert np.allclose(loaded.translation, poses.translation, atol=2e-3)
    wrapped = np.angle(np.exp(1j * (loaded.torsions - poses.torsions)))
    assert np.allclose(wrapped, 0.0, atol=2e-3)  # equal modulo 2 pi
    if rotation_type == "quat":
        assert np.all(loaded.rotation[:, 0] >= 0)  # w >= 0, the same rotation as -q


def test_the_center_atom_does_not_matter_for_the_round_trip(tmp_path):
    base = _molecule()
    other = Mol(base.rdkit, flexible=True, center_atom=[a.GetIdx() for a in base.atoms][-1])
    assert other.center_atom != base.center_atom
    poses = _poses(other, 3)
    other.to_sdf(str(tmp_path / "p.sdf"), poses)

    loaded = other.poses_from_sdf(str(tmp_path / "p.sdf"), rmsd_delta=0.01)

    assert np.allclose(other.pose_to_positions(loaded), other.pose_to_positions(poses), atol=2e-3)


def test_a_molecule_without_torsions_gives_rigid_poses(tmp_path):
    mol = _molecule(flexible=False)
    poses = _poses(mol, 3)
    mol.to_sdf(str(tmp_path / "p.sdf"), poses)

    loaded = mol.poses_from_sdf(str(tmp_path / "p.sdf"), rmsd_delta=0.01)

    assert loaded.layout == mol.layout and loaded.layout.n_tors == 0
    assert np.allclose(mol.pose_to_positions(loaded), mol.pose_to_positions(poses), atol=2e-3)


def test_rotation_from_matrix_inverts_rotation_matrix():
    for rot_type in ("euler", "quat"):
        layout = PoseLayout(rot_type, 0)
        rotations = layout.sample_random_rotations(50, np.random.default_rng(0))

        recovered = layout.rotation_from_matrix(layout.rotation_matrix(rotations))

        assert np.allclose(layout.rotation_matrix(recovered), layout.rotation_matrix(rotations))
        assert np.allclose(layout.rotation_from_matrix(np.eye(3)), layout.identity_rotation)


def test_conformers_that_a_pose_cannot_reproduce_are_rejected(tmp_path):
    mol, path = _molecule(), str(tmp_path / "bent.sdf")
    pose = _poses(mol, 1)[0]
    positions = mol.pose_to_positions(pose)
    positions[0] += [1.5, 0.0, 0.0]  # a bond stretched by a docking program, say
    work = mol.to_rdkit()
    work.GetConformer().SetPositions(positions)
    writer = Chem.SDWriter(path)
    writer.write(work)
    writer.close()

    with pytest.raises(ValueError, match="rmsd_delta"):
        mol.poses_from_sdf(path, rmsd_delta=0.05)
    assert len(mol.poses_from_sdf(path, rmsd_delta=5.0)) == 1  # accepted when allowed


def test_another_molecule_or_atom_order_is_rejected(tmp_path):
    mol, path = _molecule(), str(tmp_path / "other.sdf")
    other = Mol(Chem.MolFromSmiles("CCOCC"), flexible=True)
    other.to_sdf(path)
    with pytest.raises(ValueError, match="atoms"):
        mol.poses_from_sdf(path)

    shuffled = Chem.RenumberAtoms(mol.to_rdkit(), list(range(mol.n_atoms))[::-1])
    writer = Chem.SDWriter(path)
    writer.write(shuffled)
    writer.close()
    with pytest.raises(ValueError, match="order"):
        mol.poses_from_sdf(path)


def test_an_empty_or_missing_file_is_a_file_error(tmp_path):
    empty = tmp_path / "empty.sdf"
    empty.write_text("")

    for path in (empty, tmp_path / "missing.sdf"):
        with pytest.raises(OSError):  # RDKit's own error, not hidden behind a ValueError
            _molecule().poses_from_sdf(str(path))
