"""Small helpers shared by the tests."""

from __future__ import annotations

import numpy as np

from pyrite import Mol, Pose


def loaded_pose(mol: Mol) -> Pose:
    """The pose of `mol` as it was loaded: no rotation, its center atom where it is, its torsions."""
    layout = mol.layout
    values = np.concatenate(
        [layout.identity_rotation, mol.positions[mol._center_atom], mol.torsions]
    )
    return Pose(values, layout)
