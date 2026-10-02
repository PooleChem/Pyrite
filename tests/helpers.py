"""Small helpers shared by the tests."""

from __future__ import annotations

import numpy as np
from rdkit import Chem

from pyrite import Mol, Pose
from pyrite._util import _rotation_matrix_from_euler, _rotation_matrix_from_quat


def loaded_pose(mol: Mol) -> Pose:
    """The pose of `mol` as it was loaded: no rotation, its center atom where it is, its torsions."""
    layout = mol.layout
    values = np.concatenate(
        [layout.identity_rotation, mol.positions[mol._center_atom], mol.torsions]
    )
    return Pose(values, layout)


def rdkit_positions(mol: Mol, pose) -> np.ndarray:
    """The atom positions of a pose, computed with RDKit's own transforms.

    The independent reference for ``Mol.pose_to_positions``, which is plain numpy: start from the
    reference geometry, rotate about the center atom and translate with ``TransformConformer``
    (using the Numba rotation matrices, not the ones SciPy builds), and set every torsion with
    ``SetDihedralRad``, in that order. Accepts a pose, its raw values, or a batch of either.
    """
    values = np.asarray(pose, dtype=float)
    if values.ndim == 2:
        return np.stack([rdkit_positions(mol, row) for row in values])
    layout = mol.layout
    rotation = values[layout.rot_slice]
    rotation_matrix = (
        _rotation_matrix_from_euler if layout.rot_type == "euler" else _rotation_matrix_from_quat
    )(*rotation)[:3, :3]
    reference = mol._reference_positions

    work = mol.to_rdkit()
    conformer = work.GetConformer()
    conformer.SetPositions(reference)
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = values[layout.trans_slice] - rotation_matrix @ reference[mol.center_atom]
    Chem.rdMolTransforms.TransformConformer(conformer, transform)
    for (a, b, c, d), angle in zip(mol.rotatable_torsions, values[layout.tors_slice], strict=True):
        Chem.rdMolTransforms.SetDihedralRad(conformer, a, b, c, d, float(angle))
    return conformer.GetPositions()
