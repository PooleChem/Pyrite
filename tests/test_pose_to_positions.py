"""`Mol.pose_to_positions`: atom positions from poses in numpy, equal to what RDKit's own transforms
give, and the conformers `Mol.pose_to_conformer` makes from them."""

import numpy as np
import pytest
from helpers import rdkit_positions
from rdkit import Chem
from rdkit.Chem import AllChem

from pyrite import Mol, PoseLayout, Poses
from pyrite._util import (
    _apply_torsions,
    _pack_torsions,
    _rotation_matrix_from_euler,
    _rotation_matrix_from_quat,
)

EXAMPLES = "examples/input_files/"
# A branched molecule with nested torsions: rotating the outer ones moves the inner ones' atoms.
BRANCHED = "CCCC(CCO)(CC(=O)OC)c1ccccc1OCC"


def _from_smiles(smiles: str, rotation_type: str = "euler", seed: int = 1) -> Mol:
    rd = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(rd, randomSeed=seed)
    return Mol(Chem.RemoveHs(rd), flexible=True, rotation_type=rotation_type)


MOLECULES = {
    "factor_x": lambda rt: Mol.from_sdf(
        EXAMPLES + "factor_x_ligand.sdf", flexible=True, rotation_type=rt
    ),
    "cyp3a4": lambda rt: Mol.from_sdf(
        EXAMPLES + "cyp3a4_ligand.sdf", flexible=True, rotation_type=rt
    ),
    "biphenyl": lambda rt: _from_smiles("c1ccccc1-c1ccccc1", rt),
    "branched": lambda rt: _from_smiles(BRANCHED, rt),
    "rigid": lambda rt: Mol.from_smiles("c1ccccc1O", rotation_type=rt),
}


def _dihedral(p0, p1, p2, p3):
    """The dihedral of four points (RDKit's sign convention), vectorised over leading axes."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - np.sum(b0 * b1, axis=-1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, axis=-1, keepdims=True) * b1
    return np.arctan2(np.sum(np.cross(b1, v) * w, axis=-1), np.sum(v * w, axis=-1))


def _reference_apply_torsions(positions, quads, moving, torsions):
    """A plain numpy version of `_apply_torsions`, the reference for the numba kernel."""
    out = np.array(positions, dtype=float)
    for t, ((a, b, c, d), idx) in enumerate(zip(quads, moving, strict=True)):
        current = _dihedral(out[:, a], out[:, b], out[:, c], out[:, d])
        axis = out[:, c] - out[:, b]
        axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
        angle = (torsions[:, t] - current)[:, None, None]
        p = out[:, idx] - out[:, b][:, None, :]
        along = np.einsum("nj,nmj->nm", axis, p)[:, :, None]
        out[:, idx] = (
            out[:, b][:, None, :]
            + p * np.cos(angle)
            + np.cross(axis[:, None, :], p) * np.sin(angle)
            + axis[:, None, :] * along * (1.0 - np.cos(angle))
        )
    return out


def _random_poses(mol: Mol, n: int, seed: int = 0) -> Poses:
    rng = np.random.default_rng(seed)
    layout = mol.layout
    return Poses.from_parts(
        layout,
        layout.sample_random_rotations(n, rng),
        rng.uniform(-8, 8, (n, 3)),
        rng.uniform(-3 * np.pi, 3 * np.pi, (n, layout.n_tors)),  # also beyond ±π
    )


def _rdkit_positions(mol: Mol, poses: Poses) -> np.ndarray:
    return rdkit_positions(mol, poses)


@pytest.mark.parametrize("rot_type", ["euler", "quat"])
@pytest.mark.parametrize("name", list(MOLECULES))
def test_positions_equal_rdkit(name, rot_type):
    mol = MOLECULES[name](rot_type)
    poses = _random_poses(mol, 25)

    expected = _rdkit_positions(mol, poses)

    assert np.abs(mol.pose_to_positions(poses) - expected).max() < 1e-9


def test_a_single_pose_and_raw_arrays(name="branched"):
    mol = MOLECULES[name]("euler")
    poses = _random_poses(mol, 5)

    batch = mol.pose_to_positions(poses)

    assert batch.shape == (5, mol.n_atoms, 3)
    for i, pose in enumerate(poses):
        assert np.array_equal(mol.pose_to_positions(pose), batch[i])
        assert np.array_equal(mol.pose_to_positions(np.asarray(pose)), batch[i])
    assert np.array_equal(mol.pose_to_positions(np.asarray(poses)), batch)


def test_the_center_atom_is_placed_at_the_translation():
    mol = MOLECULES["branched"]("euler")
    poses = _random_poses(mol, 10)

    positions = mol.pose_to_positions(poses)

    assert np.allclose(positions[:, mol.center_atom], poses.translation)


def test_the_torsions_of_the_positions_are_the_requested_ones():
    mol = MOLECULES["branched"]("euler")
    poses = _random_poses(mol, 10)
    positions = mol.pose_to_positions(poses)

    for t, quad in enumerate(mol.rotatable_torsions):
        measured = _dihedral(*[positions[:, i] for i in quad])
        difference = measured - poses.torsions[:, t]
        assert np.allclose(np.angle(np.exp(1j * difference)), 0.0, atol=1e-9)  # equal modulo 2π


def test_the_order_of_the_torsions_does_not_matter():
    # Torsions are absolute and rotating about one never changes another's dihedral, so the
    # result is the same in any order, including for nested torsions.
    mol = MOLECULES["branched"]("euler")
    assert mol.n_tors >= 5
    poses = _random_poses(mol, 8)
    quads = list(mol.rotatable_torsions)
    moving = list(mol._Mol__torsion_moving)
    start = np.broadcast_to(mol._reference_positions, (8, *mol._reference_positions.shape))
    expected = _apply_torsions(start, _pack_torsions(quads, moving), poses.torsions)
    assert (
        np.abs(expected - _reference_apply_torsions(start, quads, moving, poses.torsions)).max()
        < 1e-9
    )

    for seed in range(5):
        order = np.random.default_rng(seed).permutation(len(quads))
        shuffled = _apply_torsions(
            start,
            _pack_torsions([quads[i] for i in order], [moving[i] for i in order]),
            poses.torsions[:, order],
        )
        assert np.abs(shuffled - expected).max() < 1e-9


def test_the_global_conformer_can_move_without_changing_the_poses():
    mol = MOLECULES["factor_x"]("euler")
    poses = _random_poses(mol, 6)
    before = mol.pose_to_positions(poses)

    mol.pose_to_conformer(_random_poses(mol, 1, seed=5)[0])  # moves and twists the global conformer
    after = mol.pose_to_positions(poses)

    assert np.array_equal(before, after)
    assert np.abs(after - _rdkit_positions(mol, poses)).max() < 1e-9


def test_a_changed_center_atom_still_matches_rdkit():
    mol = MOLECULES["branched"]("euler")
    mol.center_atom = 4
    poses = _random_poses(mol, 10)

    assert mol.center_atom == 4
    assert np.abs(mol.pose_to_positions(poses) - _rdkit_positions(mol, poses)).max() < 1e-9


def test_pose_to_positions_leaves_the_molecule_alone():
    mol = MOLECULES["factor_x"]("euler")
    n_conformers, positions = mol.n_conformers, mol.get_positions().copy()

    mol.pose_to_positions(_random_poses(mol, 5))

    assert mol.n_conformers == n_conformers
    assert np.array_equal(mol.get_positions(), positions)


def test_wrong_layout_or_size_is_rejected():
    mol = MOLECULES["factor_x"]("euler")
    other = MOLECULES["biphenyl"]("euler")

    with pytest.raises(AssertionError):
        mol.pose_to_positions(_random_poses(other, 2))
    with pytest.raises(ValueError):
        mol.pose_to_positions(np.zeros(mol.layout.n_dims + 1))


@pytest.mark.parametrize("rot_type", ["euler", "quat"])
def test_layout_rotation_matrix_matches_the_numba_matrices(rot_type):
    layout = PoseLayout(rot_type, 0)
    rotations = layout.sample_random_rotations(50, np.random.default_rng(0))
    f = _rotation_matrix_from_euler if rot_type == "euler" else _rotation_matrix_from_quat

    matrices = layout.rotation_matrix(rotations)

    assert matrices.shape == (50, 3, 3)
    for r, m in zip(rotations, matrices, strict=True):
        assert np.allclose(m, f(*r)[:3, :3], atol=1e-12)
    assert np.allclose(layout.rotation_matrix(rotations[0]), matrices[0])


def test_an_empty_batch_has_no_positions():
    mol = MOLECULES["factor_x"]("euler")
    empty = _random_poses(mol, 3)[:0]

    assert mol.pose_to_positions(empty).shape == (0, mol.n_atoms, 3)


def test_a_changed_center_atom_keeps_the_center_atom_at_the_translation():
    # The center atom must be on the fixed side of every torsion, or a pose's translation is not
    # where the center atom ends up. This used to hold only for a center atom given to the
    # constructor, not for one set afterwards.
    for name in ("factor_x", "branched"):
        mol = MOLECULES[name]("euler")
        torsions = np.array(mol.torsions).copy()
        mol.center_atom = 4
        poses = _random_poses(mol, 12)

        positions = mol.pose_to_positions(poses)

        assert np.allclose(positions[:, 4], poses.translation, atol=1e-9), name
        assert np.abs(positions - _rdkit_positions(mol, poses)).max() < 1e-9
        # the torsions themselves are unchanged (reversing a torsion gives the same angle)
        assert np.allclose(np.angle(np.exp(1j * (np.array(mol.torsions) - torsions))), 0, atol=1e-9)


def test_setting_the_center_atom_resets_the_orientation_and_moves_it_to_the_origin():
    mol = MOLECULES["factor_x"]("euler")
    mol.pose_to_conformer(_random_poses(mol, 1, seed=3)[0])  # rotate, move and twist it first
    torsions = np.array(mol.torsions).copy()
    internal = Chem.Get3DDistanceMatrix(mol.rdkit)

    mol.center_atom = 4

    assert np.allclose(mol.get_positions()[4], 0.0, atol=1e-9)
    assert np.allclose(mol.rotation, mol.layout.identity_rotation)
    assert np.allclose(Chem.Get3DDistanceMatrix(mol.rdkit), internal, atol=1e-9)  # not deformed
    assert np.allclose(np.angle(np.exp(1j * (np.array(mol.torsions) - torsions))), 0, atol=1e-9)


@pytest.mark.parametrize("rot_type", ["euler", "quat"])
@pytest.mark.parametrize("name", ["factor_x", "branched", "rigid"])
def test_pose_to_conformer_puts_the_positions_on_the_conformer(name, rot_type):
    mol = MOLECULES[name](rot_type)
    poses = _random_poses(mol, 4)
    loaded = mol.get_positions().copy()

    new_ids = [mol.pose_to_conformer(p, new_conf=True) for p in poses]

    assert len(set(new_ids)) == 4 and -1 not in new_ids
    assert np.array_equal(mol.get_positions(), loaded)  # a new conformer leaves the global one
    for conf_id, expected in zip(new_ids, mol.pose_to_positions(poses), strict=True):
        assert np.array_equal(mol.get_positions(conf_id), expected)
        assert mol.rdkit.GetConformer(conf_id).Is3D()
        mol.remove_conformer(conf_id)

    assert mol.pose_to_conformer(poses[0]) == -1  # without `new_conf`: the global conformer
    assert np.array_equal(mol.get_positions(), mol.pose_to_positions(poses[0]))
    assert np.array_equal(mol.rotation, poses[0].rotation)
    assert mol.n_conformers == 1


def test_the_pose_does_not_depend_on_how_the_conformer_got_there():
    # Applying poses one after the other gives the same conformer as applying the last one alone.
    mol = MOLECULES["branched"]("euler")
    poses = _random_poses(mol, 6)
    for pose in poses:
        mol.pose_to_conformer(pose)

    assert np.array_equal(mol.get_positions(), mol.pose_to_positions(poses[-1]))
    assert np.abs(mol.get_positions() - rdkit_positions(mol, poses[-1])).max() < 1e-9
