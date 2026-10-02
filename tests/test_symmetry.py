"""Symmetry-aware RMSD: the `RMSD` and `Crowding` scoring functions, and `pyrite.cluster`."""

import numpy as np
import pytest
from helpers import loaded_pose
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from pyrite import Mol
from pyrite._common import Poses
from pyrite.cluster import cluster_and_select, rmsd_matrix
from pyrite.scoring import RMSD, Crowding


def _mol(smiles: str, seed: int = 1) -> Mol:
    """Build a heavy-atom Mol with one embedded conformer."""
    rd = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(rd, randomSeed=seed)
    return Mol(Chem.RemoveHs(rd), flexible=True)


def _relabelled(mol: Mol, sigma) -> Mol:
    """Copy `mol` with atom i placed where atom sigma[i] is: the same structure, relabelled."""
    copy = Mol(mol.rdkit)
    positions = mol.get_positions()
    conformer = copy.rdkit.GetConformer()
    for i, j in enumerate(sigma):
        conformer.SetAtomPosition(i, tuple(float(x) for x in positions[j]))
    return copy


def _most_mobile_symmetry(mol: Mol):
    automorphisms = mol.rdkit.GetSubstructMatches(mol.rdkit, uniquify=False, useChirality=True)
    return max(automorphisms, key=lambda s: sum(i != j for i, j in enumerate(s)))


@pytest.fixture(scope="module")
def biphenyl() -> Mol:
    return _mol("c1ccccc1-c1ccccc1")


def test_rmsd_is_zero_for_a_symmetric_relabelling(biphenyl):
    sigma = _most_mobile_symmetry(biphenyl)
    relabelled = _relabelled(biphenyl, sigma)
    identity = [[(i, i) for i in range(biphenyl.n_atoms)]]

    assert rdMolAlign.CalcRMS(biphenyl.rdkit, relabelled.rdkit, map=identity) > 1.0
    assert RMSD(biphenyl, relabelled).get_score(loaded_pose(biphenyl)) < 1e-6


def test_rmsd_does_not_depend_on_the_atom_order_of_the_reference():
    mol = _mol("CC(N)C(=O)OC")
    order = [int(i) for i in np.random.default_rng(0).permutation(mol.n_atoms)]
    renumbered = Mol(Chem.RenumberAtoms(mol.rdkit, order))

    assert RMSD(mol, renumbered).get_score(loaded_pose(mol)) < 1e-6


def test_rmsd_equals_rdkit_for_different_conformers(biphenyl):
    other = _mol("c1ccccc1-c1ccccc1", seed=7)

    assert RMSD(biphenyl, other).get_score(loaded_pose(biphenyl)) == pytest.approx(
        rdMolAlign.CalcRMS(biphenyl.rdkit, other.rdkit), abs=1e-6
    )


def test_crowding_recognises_a_symmetric_duplicate(biphenyl):
    crowding = Crowding(biphenyl, offset=4.0)
    relabelled = _relabelled(biphenyl, _most_mobile_symmetry(biphenyl))
    conf_id = crowding._ref_mol.rdkit.AddConformer(relabelled.rdkit.GetConformer(), assignId=True)
    crowding.register_pose(int(conf_id))

    assert crowding.get_score(loaded_pose(biphenyl)) == pytest.approx(4**4.0, rel=1e-6)


def _poses(mol: Mol, n: int) -> Poses:
    rng = np.random.default_rng(0)
    rotation = rng.uniform(-1, 1, (n, 3))
    translation = rng.uniform(-3, 3, (n, 3))
    torsions = rng.uniform(-np.pi, np.pi, (n, mol.n_tors))
    return Poses(np.concatenate([rotation, translation, torsions], axis=1), mol.layout)


def test_rmsd_matrix_agrees_with_the_rmsd_scoring_function(biphenyl):
    poses = _poses(biphenyl, 6)
    matrix = rmsd_matrix(biphenyl, poses)
    probe, ref = Mol(biphenyl.rdkit, flexible=True), Mol(biphenyl.rdkit, flexible=True)

    for i, j in [(0, 1), (2, 5), (3, 4)]:
        ref.pose_to_conformer(poses[i])
        assert RMSD(probe, ref).get_score(poses[j]) == pytest.approx(matrix[i, j], abs=1e-6)


def test_rmsd_matrix_symmetry_never_exceeds_the_plain_rmsd(biphenyl):
    poses = _poses(biphenyl, 6)
    plain = rmsd_matrix(biphenyl, poses, symmetry=False)

    assert (rmsd_matrix(biphenyl, poses) <= plain + 1e-9).all()
    with pytest.warns(UserWarning, match="symmetry-equivalent"):
        capped = rmsd_matrix(biphenyl, poses, max_matches=2)
    assert (capped <= plain + 1e-9).all()


def test_cluster_and_select_merges_symmetric_duplicates(biphenyl):
    poses = _poses(biphenyl, 2)
    sigma = _most_mobile_symmetry(biphenyl)
    positions = biphenyl.get_positions()
    scores = np.array([1.0, 2.0])

    # Pose 1 is pose 0 with the atoms relabelled by a symmetry: the same structure.
    from unittest import mock

    stack = np.stack([positions, positions[list(sigma)]])
    with mock.patch.object(Mol, "pose_to_positions", lambda mol, poses: stack):
        assert list(cluster_and_select(biphenyl, poses, scores, cutoff=0.5)) == [0]
        assert sorted(cluster_and_select(biphenyl, poses, scores, cutoff=0.5, symmetry=False)) == [
            0,
            1,
        ]
