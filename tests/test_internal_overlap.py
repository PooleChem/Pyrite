"""`InternalOverlap`: the overlap of atoms more than 4 bonds apart, in numpy, for any batch."""

from unittest import mock

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from pyrite import Mol, Poses
from pyrite.scoring import InternalOverlap


def _reference(mol: Mol, pose, vdw_scale: float = 1.0) -> float:
    """The overlap by the book, with RDKit: its 3D distance matrix and its topological one."""
    table = Chem.GetPeriodicTable()
    radii = np.array([table.GetRvdw(a.GetAtomicNum()) * vdw_scale for a in mol.atoms])
    distances = Chem.Get3DDistanceMatrix(mol.to_rdkit(pose))
    topological = Chem.GetDistanceMatrix(mol.rdkit)
    total = 0.0
    for i in range(mol.n_atoms):
        for j in range(i + 1, mol.n_atoms):
            if topological[i, j] > 4 and mol.scoring_mask[i] and mol.scoring_mask[j]:
                total += max(radii[i] + radii[j] - distances[i, j], 0.0)
    return total


def _with_hydrogens(**kwargs) -> Mol:
    rd = Chem.AddHs(Chem.MolFromSmiles("CCCC(CCO)(CC(=O)OC)c1ccccc1OCC"))
    AllChem.EmbedMolecule(rd, randomSeed=1)
    return Mol(rd, hydrogens="keep", flexible=True, **kwargs)


def _poses(mol: Mol, n: int, seed: int = 0) -> Poses:
    layout, rng = mol.layout, np.random.default_rng(seed)
    return Poses.from_parts(
        layout,
        layout.sample_random_rotations(n, rng),
        rng.uniform(-5, 5, (n, 3)),
        rng.uniform(-np.pi, np.pi, (n, layout.n_tors)),
    )


@pytest.mark.parametrize("which", ["ligand", "hydrogens ignored", "hydrogens counted"])
def test_equals_the_reference_for_one_pose_and_for_a_batch(ctx, which):
    mol = {
        "ligand": lambda: ctx.ligand,
        "hydrogens ignored": _with_hydrogens,
        "hydrogens counted": lambda: _with_hydrogens(ignore_hydrogens=False),
    }[which]()
    poses = _poses(mol, 25)
    term = InternalOverlap(mol)

    expected = np.array([_reference(mol, p) for p in poses])

    assert expected.max() > 0.5  # the poses overlap: the test is not about zeros
    assert np.allclose([term.get_score(p) for p in poses], expected, atol=1e-9)
    assert np.allclose(term.batch_scores(poses), expected, atol=1e-9)


def test_the_molecules_scoring_mask_decides_which_atoms_take_part():
    ignoring, counting = _with_hydrogens(), _with_hydrogens(ignore_hydrogens=False)
    poses = _poses(ignoring, 15)

    ignored = InternalOverlap(ignoring).batch_scores(poses)
    counted = InternalOverlap(counting).batch_scores(poses)

    # the default leaves every pair with a hydrogen out, so hydrogens can only add overlap
    assert np.all(ignored <= counted + 1e-12) and ignored.max() < counted.max()
    assert not InternalOverlap(ignoring)._first.size == InternalOverlap(counting)._first.size
    heavy = np.array([a.GetAtomicNum() > 1 for a in ignoring.atoms])
    term = InternalOverlap(ignoring)
    assert heavy[term._first].all() and heavy[term._second].all()


def test_the_radii_scale(ctx):
    poses = _poses(ctx.ligand, 10)
    scores = {
        s: InternalOverlap(ctx.ligand, vdw_scale=s).batch_scores(poses)
        for s in (0.0, 0.5, 1.0, 1.5)
    }

    assert np.all(scores[0.0] == 0.0)
    assert np.all(scores[0.5] <= scores[1.0]) and np.all(scores[1.0] <= scores[1.5])
    expected = np.array([_reference(ctx.ligand, p, vdw_scale=1.5) for p in poses])
    assert np.allclose(scores[1.5], expected, atol=1e-9)


def test_it_depends_only_on_the_torsions(ctx):
    mol = ctx.ligand
    term = InternalOverlap(mol)
    base = np.array(_poses(mol, 1)[0])
    moved = base.copy()
    layout = mol.layout
    moved[layout.rot_slice] = layout.sample_random_rotations(1, np.random.default_rng(3))[0]
    moved[layout.trans_slice] += [4.0, -3.0, 7.0]

    assert term.get_score(moved) == pytest.approx(term.get_score(base), abs=1e-9)


def test_scoring_uses_no_rdkit_and_no_conformer(ctx):
    term = InternalOverlap(ctx.ligand)
    poses = _poses(ctx.ligand, 5)

    with (
        mock.patch.object(Mol, "to_rdkit", side_effect=AssertionError("RDKit used")),
        mock.patch.object(Mol, "pose_to_conformer", side_effect=AssertionError("conformer made")),
    ):
        term.get_score(poses[0])
        term.batch_scores(poses)
