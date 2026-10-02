"""Sampling of poses: `PoseLayout.sample_random_*`, `Poses.from_parts`, and shape validation."""

import numpy as np
import pytest
from helpers import rdkit_positions
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy import stats

from pyrite import Mol
from pyrite._common import Pose, PoseLayout, Poses
from pyrite._util import _rotation_matrix_from_euler, _rotation_matrix_from_quat

N = 20000


def _matrices(layout: PoseLayout, rotations: np.ndarray) -> np.ndarray:
    matrix = (
        _rotation_matrix_from_euler if layout.rot_type == "euler" else _rotation_matrix_from_quat
    )
    return np.stack([matrix(*r)[:3, :3] for r in rotations])


def _z_image_ks_pvalue(matrices: np.ndarray) -> float:
    """Images of a fixed vector under uniform rotations are uniform on the sphere: z is uniform."""
    z = (matrices @ np.array([0.0, 0.0, 1.0]))[:, 2]
    return stats.kstest((z + 1) / 2, "uniform").pvalue


@pytest.mark.parametrize("rot_type", ["euler", "quat"])
def test_random_rotations_are_uniform_on_so3(rot_type):
    layout = PoseLayout(rot_type, 0)
    rotations = layout.sample_random_rotations(N, np.random.default_rng(0))
    matrices = _matrices(layout, rotations)

    assert rotations.shape == (N, layout.rot_dim)
    assert np.allclose(np.linalg.det(matrices), 1.0)
    assert np.allclose(matrices @ matrices.transpose(0, 2, 1), np.eye(3), atol=1e-9)
    assert _z_image_ks_pvalue(matrices) > 1e-3
    assert np.abs(matrices.mean(axis=0)).max() < 0.03


def test_uniform_euler_angles_are_not_uniform_on_so3():
    # What the placement used to do. Guards the sensitivity of the test above.
    angles = np.random.default_rng(0).uniform(-np.pi, np.pi, (N, 3))
    matrices = _matrices(PoseLayout("euler", 0), angles)

    assert _z_image_ks_pvalue(matrices) < 1e-6


def test_random_quaternions_have_unit_norm():
    rotations = PoseLayout("quat", 0).sample_random_rotations(100, np.random.default_rng(1))

    assert np.allclose(np.linalg.norm(rotations, axis=1), 1.0)


def test_euler_and_quat_layouts_describe_the_same_rotation():
    rd = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1-c1ccccc1"))
    AllChem.EmbedMolecule(rd, randomSeed=1)
    euler = Mol(Chem.Mol(rd), flexible=True, rotation_type="euler")
    quat = Mol(Chem.Mol(rd), flexible=True, rotation_type="quat")
    quaternions = quat.layout.sample_random_rotations(5, np.random.default_rng(3))
    eulers = Mol(Chem.Mol(rd), rotation_type="euler").layout.sample_random_rotations(
        5, np.random.default_rng(3)
    )
    translation, torsion = np.array([1.0, -2.0, 3.0]), np.array([0.3])

    for q, e in zip(quaternions, eulers, strict=True):
        a = rdkit_positions(quat, np.concatenate([q, translation, torsion]))
        b = rdkit_positions(euler, np.concatenate([e, translation, torsion]))
        assert np.allclose(a, b, atol=1e-9)
        assert np.allclose(
            quat.pose_to_positions(np.concatenate([q, translation, torsion])), a, atol=1e-9
        )


@pytest.mark.parametrize("rot_type", ["euler", "quat"])
def test_sampling_is_seeded(rot_type):
    layout = PoseLayout(rot_type, 3)

    a = layout.sample_random_rotations(10, np.random.default_rng(5))
    b = layout.sample_random_rotations(10, np.random.default_rng(5))
    c = layout.sample_random_rotations(10, np.random.default_rng(6))
    assert np.array_equal(a, b) and not np.array_equal(a, c)

    a = layout.sample_random_torsions(10, np.random.default_rng(5))
    b = layout.sample_random_torsions(10, np.random.default_rng(5))
    assert np.array_equal(a, b)


def test_random_torsions_shape_and_range():
    torsions = PoseLayout("euler", 4).sample_random_torsions(1000, np.random.default_rng(0))

    assert torsions.shape == (1000, 4)
    assert torsions.min() >= -np.pi and torsions.max() < np.pi
    assert PoseLayout("euler", 0).sample_random_torsions(7).shape == (7, 0)


def test_pose_and_poses_validate_their_shape():
    layout = PoseLayout("euler", 2)  # 3 + 3 + 2 = 8 variables

    Pose(np.zeros(8), layout)
    Poses(np.zeros((5, 8)), layout)
    with pytest.raises(ValueError):
        Pose(np.zeros(9), layout)
    with pytest.raises(ValueError):
        Pose(np.zeros((1, 8)), layout)
    with pytest.raises(ValueError):
        Poses(np.zeros((5, 7)), layout)
    with pytest.raises(ValueError):
        Poses(np.zeros(8), layout)


def test_from_parts_round_trip():
    layout = PoseLayout("quat", 2)
    rng = np.random.default_rng(0)
    rotation = layout.sample_random_rotations(6, rng)
    translation = rng.normal(size=(6, 3))
    torsions = layout.sample_random_torsions(6, rng)

    poses = Poses.from_parts(layout, rotation, translation, torsions)

    assert len(poses) == 6 and poses.layout == layout
    assert np.array_equal(poses.rotation, rotation)
    assert np.array_equal(poses.translation, translation)
    assert np.array_equal(poses.torsions, torsions)


def test_poses_follow_the_numpy_array_protocol():
    # NumPy 2 passes `copy`; the owned array is returned as is (no allocation) unless a copy is asked.
    layout = PoseLayout("euler", 2)
    poses = Poses(np.zeros((3, layout.n_dims)), layout)
    pose = poses[0]

    assert np.asarray(poses) is poses._vs
    assert np.asarray(pose) is pose._v
    assert np.array(poses) is not poses._vs and np.array(pose) is not pose._v
    assert np.asarray(poses, dtype=np.float32).dtype == np.float32
    with pytest.raises(ValueError):
        np.array(poses, dtype=np.float32, copy=False)


def test_poses_from_list_stacks_poses_and_keeps_the_layout():
    layout = PoseLayout("quat", 2)
    rng = np.random.default_rng(0)
    poses = [Pose(rng.normal(size=layout.n_dims), layout) for _ in range(4)]

    stacked = Poses.from_list(poses)

    assert stacked.layout == layout and len(stacked) == 4
    assert np.array_equal(np.asarray(stacked), np.stack([np.asarray(p) for p in poses]))
    assert Poses.from_list(poses, layout) == stacked
    with pytest.raises((ValueError, AssertionError)):
        Poses.from_list(
            [poses[0], Pose(np.zeros(PoseLayout("euler", 2).n_dims), PoseLayout("euler", 2))]
        )
