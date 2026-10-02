"""Regression tests for `pyrite.cluster`: the RMSD matrix and the selection of cluster representatives."""

from unittest import mock

import numpy as np
import pytest
from helpers import rdkit_positions
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from rdkit.ML.Cluster import Butina

from pyrite import Mol
from pyrite._common import Poses
from pyrite.cluster import cluster_and_select, rmsd_matrix


@pytest.fixture(scope="module")
def mol() -> Mol:
    rd = Chem.AddHs(Chem.MolFromSmiles("CC(N)C(=O)Oc1ccccc1"))  # flexible, with a symmetric ring
    AllChem.EmbedMolecule(rd, randomSeed=2)
    return Mol(Chem.RemoveHs(rd), flexible=True)


def _poses(mol: Mol, n_groups: int = 6, per_group: int = 5, seed: int = 0) -> Poses:
    """Groups of near-identical poses, so that there are real clusters to find."""
    rng = np.random.default_rng(seed)
    layout = mol.layout
    base = Poses.from_parts(
        layout,
        layout.sample_random_rotations(n_groups, rng),
        rng.uniform(-6, 6, (n_groups, 3)),
        layout.sample_random_torsions(n_groups, rng),
    )
    jitter = rng.normal(scale=0.02, size=(n_groups * per_group, layout.n_dims))
    return Poses(np.repeat(np.asarray(base), per_group, axis=0) + jitter, layout)


def _calc_rms_matrix(mol: Mol, poses: Poses) -> np.ndarray:
    """The RMSD matrix from RDKit's own CalcRMS, one conformer per pose (an independent path)."""
    work = mol.to_rdkit()
    work.RemoveAllConformers()
    for pose in poses:
        conformer = Chem.Conformer(mol.rdkit.GetConformer())
        conformer.SetPositions(rdkit_positions(mol, pose))
        work.AddConformer(conformer, assignId=True)
    n_atoms = work.GetNumAtoms()
    matches = work.GetSubstructMatches(work, uniquify=False, useChirality=True, maxMatches=100000)
    atom_maps = [[(probe, ref) for ref, probe in enumerate(m)] for m in matches]
    n = len(poses)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j] = matrix[j, i] = rdMolAlign.CalcRMS(
                work, work, prbId=j, refId=i, map=atom_maps
            )
    assert n_atoms == mol.n_atoms
    return matrix


def test_rmsd_matrix_agrees_with_rdkit(mol):
    poses = _poses(mol, n_groups=4, per_group=3)

    assert np.allclose(rmsd_matrix(mol, poses), _calc_rms_matrix(mol, poses), atol=1e-8)


def test_rmsd_matrix_is_a_metric(mol):
    poses = _poses(mol)
    matrix = rmsd_matrix(mol, poses)
    n = len(poses)

    assert matrix.shape == (n, n)
    assert np.allclose(matrix, matrix.T) and (np.diag(matrix) == 0).all() and (matrix >= 0).all()
    rng = np.random.default_rng(0)
    for i, j, k in rng.integers(0, n, (200, 3)):
        assert matrix[i, k] <= matrix[i, j] + matrix[j, k] + 1e-9


def test_rmsd_matrix_leaves_the_molecule_and_poses_alone(mol):
    poses = _poses(mol)
    n_conformers, positions = mol.n_conformers, mol.get_positions().copy()
    values = np.asarray(poses).copy()

    rmsd_matrix(mol, poses)

    assert mol.n_conformers == n_conformers
    assert np.array_equal(mol.get_positions(), positions)
    assert np.array_equal(np.asarray(poses), values)


def test_rmsd_matrix_edge_cases(mol):
    poses = _poses(mol)

    assert rmsd_matrix(mol, poses[:0]).shape == (0, 0)
    assert rmsd_matrix(mol, poses[:1]).tolist() == [[0.0]]
    broken = np.asarray(poses).copy()
    broken[3, 0] = np.nan
    with pytest.raises(ValueError):
        rmsd_matrix(mol, Poses(broken, mol.layout))


def test_cluster_and_select_returns_the_best_pose_of_every_cluster(mol):
    poses = _poses(mol)
    scores = np.random.default_rng(1).normal(size=len(poses))
    cutoff = 0.5

    selected = cluster_and_select(mol, poses, scores, cutoff=cutoff)

    # The groups of near-identical poses are the clusters: one representative per group, and
    # it is the group's best-scoring pose.
    groups = np.arange(len(poses)) // 5
    assert len(selected) == 6 and sorted(groups[selected]) == list(range(6))
    for group in range(6):
        members = np.where(groups == group)[0]
        assert members[np.argmin(scores[members])] in selected
    assert (np.diff(scores[selected]) >= 0).all() and len(set(selected.tolist())) == len(selected)
    # ... and these are exactly the clusters Butina finds on the RDKit reference matrix.
    clusters = Butina.ClusterData(
        _calc_rms_matrix(mol, poses), len(poses), cutoff, isDistData=True, reordering=True
    )
    expected = sorted(min(c, key=lambda i: scores[i]) for c in clusters)
    assert sorted(selected.tolist()) == expected


def test_cluster_and_select_cutoff_extremes_and_truncation(mol):
    poses = _poses(mol)
    scores = np.random.default_rng(2).normal(size=len(poses))

    assert list(cluster_and_select(mol, poses, scores, cutoff=1e6)) == [int(np.argmin(scores))]
    assert len(cluster_and_select(mol, poses, scores, cutoff=1e-9)) == len(poses)
    everything = cluster_and_select(mol, poses, scores, cutoff=0.5)
    for k in (1, 3, 100):
        assert np.array_equal(
            cluster_and_select(mol, poses, scores, cutoff=0.5, n_output=k), everything[:k]
        )


def test_cluster_and_select_edge_cases_and_validation(mol):
    poses = _poses(mol)

    assert cluster_and_select(mol, poses[:0], np.zeros(0)).shape == (0,)
    assert list(cluster_and_select(mol, poses[:1], np.array([3.0]))) == [0]
    with pytest.raises(ValueError):
        cluster_and_select(mol, poses, np.zeros(len(poses) - 1))
    with pytest.raises(ValueError):
        cluster_and_select(mol, poses, np.zeros(len(poses)), n_output=0)


def test_cluster_and_select_uses_butina_with_reordering(mol):
    # Chained points on a line: here Butina's `reordering` (update the neighbour counts of the
    # unassigned points after every cluster) changes which poses are selected.
    def representatives(clusters, scores):
        return sorted(min(c, key=lambda i: scores[i]) for c in clusters)

    rng = np.random.default_rng(0)
    for _ in range(500):
        x = np.sort(rng.uniform(0, 12, 14))
        distances = np.abs(x[:, None] - x[None])
        scores = rng.normal(size=14)
        with_reordering = Butina.ClusterData(distances, 14, 1.0, isDistData=True, reordering=True)
        without = Butina.ClusterData(distances, 14, 1.0, isDistData=True, reordering=False)
        if representatives(with_reordering, scores) != representatives(without, scores):
            break
    else:
        pytest.fail("no example found where the reordering matters")
    poses = Poses(np.zeros((14, mol.layout.n_dims)), mol.layout)

    with mock.patch("pyrite.cluster.rmsd_matrix", lambda *args, **kwargs: distances):
        selected = cluster_and_select(mol, poses, scores, cutoff=1.0)

    assert sorted(selected.tolist()) == representatives(with_reordering, scores)
