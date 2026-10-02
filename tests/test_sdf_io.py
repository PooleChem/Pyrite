"""Writing poses to an SDF file and reading them back: `Mol.v_to_sdf` and `Mol.v_from_sdf`."""

import numpy as np
from rdkit import Chem

from pyrite import Mol, Poses


def _poses(mol: Mol, n: int, seed: int = 5) -> Poses:
    layout, rng = mol.layout, np.random.default_rng(seed)
    return Poses.from_parts(
        layout,
        layout.sample_random_rotations(n, rng),
        rng.uniform(-5, 5, (n, 3)),
        rng.uniform(-np.pi, np.pi, (n, layout.n_tors)),
    )


def test_poses_written_to_sdf_are_the_positions_of_the_poses(ctx, tmp_path):
    ligand, poses = ctx.ligand, _poses(ctx.ligand, 4)
    path = str(tmp_path / "poses.sdf")

    ligand.v_to_sdf(path, poses)

    written = [m.GetConformer().GetPositions() for m in Chem.SDMolSupplier(path, sanitize=False)]
    assert len(written) == 4
    assert ligand.n_conformers == 1  # the molecule is left as it was
    # an SDF file has 4 decimals
    assert np.allclose(written, ligand.pose_to_positions(poses), atol=1e-3)


def test_poses_survive_a_round_trip_through_an_sdf_file(ctx, tmp_path):
    ligand, poses = ctx.ligand.copy(), _poses(ctx.ligand, 3)
    path = str(tmp_path / "poses.sdf")
    ligand.v_to_sdf(path, poses)

    loaded, variables = Mol.v_from_sdf(path, flexible=True, rmsd_delta=2.0)

    assert len(variables) == 3 and loaded.n_tors == ligand.n_tors
    written = [m.GetConformer().GetPositions() for m in Chem.SDMolSupplier(path, sanitize=False)]
    # The recovered poses, applied to the molecule that was loaded, give the written positions.
    for values, expected in zip(variables, written, strict=True):
        assert np.allclose(loaded.pose_to_positions(np.array(values)), expected, atol=2e-3)
