"""Symmetry-aware RMSD: the `RMSD` and `Crowding` scoring functions, and `pyrite.cluster`."""

import numpy as np
import pytest
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
    assert RMSD(biphenyl, relabelled).get_score(biphenyl.input_pose) < 1e-6


def test_rmsd_does_not_depend_on_the_atom_order_of_the_reference():
    mol = _mol("CC(N)C(=O)OC")
    order = [int(i) for i in np.random.default_rng(0).permutation(mol.n_atoms)]
    renumbered = Mol(Chem.RenumberAtoms(mol.rdkit, order))

    assert RMSD(mol, renumbered).get_score(mol.input_pose) < 1e-6


def test_rmsd_equals_rdkit_for_different_conformers(biphenyl):
    other = _mol("c1ccccc1-c1ccccc1", seed=7)

    assert RMSD(biphenyl, other).get_score(biphenyl.input_pose) == pytest.approx(
        rdMolAlign.CalcRMS(biphenyl.rdkit, other.rdkit), abs=1e-6
    )


def test_crowding_recognises_a_symmetric_duplicate(biphenyl):
    crowding = Crowding(biphenyl, offset=4.0)
    relabelled = _relabelled(biphenyl, _most_mobile_symmetry(biphenyl))
    conf_id = crowding._ref_mol.rdkit.AddConformer(relabelled.rdkit.GetConformer(), assignId=True)
    crowding.register_pose(int(conf_id))

    assert crowding.get_score(biphenyl.input_pose) == pytest.approx(4**4.0, rel=1e-6)


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


# ---------------------------------------------------------------------------
# Heavy atoms only: the hydrogens a Mol was loaded with do not change the RMSD
# ---------------------------------------------------------------------------


def _with_hydrogens(smiles: str, seed: int = 1) -> Mol:
    """A Mol that keeps all its hydrogens, with one embedded conformer."""
    rd = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(rd, randomSeed=seed)
    return Mol(rd, flexible=True, hydrogens="keep")


def _heavy_rms(probe: Chem.Mol, ref: Chem.Mol) -> float:
    """RDKit's own symmetry-aware RMSD over hydrogen-free copies (an independent path)."""
    return rdMolAlign.CalcRMS(
        Chem.RemoveAllHs(probe, sanitize=False), Chem.RemoveAllHs(ref, sanitize=False)
    )


def test_moving_only_hydrogens_does_not_change_the_rmsd():
    mol = _with_hydrogens("CC(=O)Nc1ccc(O)cc1")
    ref = Mol(mol.rdkit, flexible=True, hydrogens="keep")
    conformer = ref.rdkit.GetConformer()
    for atom in ref.atoms:
        if atom.GetAtomicNum() == 1:
            p = conformer.GetAtomPosition(atom.GetIdx())
            conformer.SetAtomPosition(atom.GetIdx(), (p.x + 0.7, p.y - 0.4, p.z + 0.5))

    assert rdMolAlign.CalcRMS(mol.rdkit, ref.rdkit) > 0.2  # all atoms: they did move
    assert RMSD(mol, ref).get_score(mol.input_pose) < 1e-9


@pytest.mark.parametrize("smiles", ["CC(=O)Nc1ccc(O)cc1", "CC(C)(C)c1ccc(C(C)(C)C)cc1"])
def test_rmsd_and_rmsd_matrix_are_heavy_atom_rmsds(smiles):
    # The second has six methyl groups: (3!)^6 hydrogen swaps on top of the heavy-atom symmetries,
    # far over max_matches when hydrogens count, a handful when they do not (so no warning).
    mol = _with_hydrogens(smiles)
    poses = _poses(mol, 5)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        matrix = rmsd_matrix(mol, poses)
        rmsd = RMSD(mol)
        scores = rmsd.batch_scores(poses)

    crystal = mol.rdkit
    for i in range(5):
        posed_i = mol.to_rdkit(poses[i])
        assert scores[i] == pytest.approx(_heavy_rms(posed_i, crystal), abs=1e-6)
        for j in range(i + 1, 5):
            assert matrix[i, j] == pytest.approx(
                _heavy_rms(posed_i, mol.to_rdkit(poses[j])), abs=1e-6
            )


def test_the_rmsd_is_the_same_whichever_hydrogens_are_kept():
    # The same geometry loaded with all, polar or no hydrogens gives the same RMSD to the crystal.
    rd = Chem.AddHs(Chem.MolFromSmiles("OCC(=O)Nc1ccc(C)cc1"))
    AllChem.EmbedMolecule(rd, randomSeed=3)
    moved = Chem.Mol(rd)
    AllChem.EmbedMolecule(moved, randomSeed=11)
    scores = []
    for hydrogens in ("keep", "polar", "remove"):
        crystal = Mol(rd, flexible=True, hydrogens=hydrogens)
        probe = Mol(moved, flexible=True, hydrogens=hydrogens)
        scores.append(RMSD(probe, crystal).get_score(probe.input_pose))

    assert scores == pytest.approx([scores[0]] * 3, abs=1e-9)
    assert scores[0] > 0.1


def test_hydrogens_between_the_heavy_atoms_are_skipped_correctly():
    # AddHs puts every hydrogen after the heavy atoms; files often interleave them. Interleaved,
    # the k-th heavy atom is no longer atom k.
    rd = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1"))
    AllChem.EmbedMolecule(rd, randomSeed=1)
    heavy = [a.GetIdx() for a in rd.GetAtoms() if a.GetAtomicNum() > 1]
    hydrogens = [a.GetIdx() for a in rd.GetAtoms() if a.GetAtomicNum() == 1]
    order = [i for pair in zip(hydrogens, heavy, strict=False) for i in pair]
    order += heavy[len(hydrogens) :] + hydrogens[len(heavy) :]
    mol = Mol(Chem.RenumberAtoms(rd, order), flexible=True, hydrogens="keep")
    assert mol.rdkit.GetAtomWithIdx(0).GetAtomicNum() == 1
    poses = _poses(mol, 4)

    scores = RMSD(mol).batch_scores(poses)
    matrix = rmsd_matrix(mol, poses)

    for i in range(4):
        posed_i = mol.to_rdkit(poses[i])
        assert scores[i] == pytest.approx(_heavy_rms(posed_i, mol.rdkit), abs=1e-6)
        assert matrix[0, i] == pytest.approx(_heavy_rms(mol.to_rdkit(poses[0]), posed_i), abs=1e-6)


# ---------------------------------------------------------------------------
# The same mappings as RDKit's CalcRMS: conjugated terminal groups, no chirality
# ---------------------------------------------------------------------------


def _from_sdf(smiles: str, tmp_path, seed: int = 1) -> Mol:
    """Load through an SDF, as docking inputs are: chiral tags are then perceived from 3D."""
    rd = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(rd, randomSeed=seed)
    path = str(tmp_path / "mol.sdf")
    with Chem.SDWriter(path) as writer:
        writer.write(rd)
    return Mol.from_sdf(path, flexible=True, hydrogens="keep")


@pytest.mark.parametrize(
    "smiles",
    [
        "OC(=O)Cc1cccc2ccccc12",  # a carboxylic acid: its two O are interchangeable
        "CC(C)(CO)[C@@H](O)C(=O)[O-]",  # a carboxylate drawn with one O charged (Astex 1n2j)
        "NC(=N)c1ccc(C)cc1",  # an amidine: its two N are interchangeable
        "NS(=O)(=O)c1ccc2c(c1)CNCC2",  # a sulfonyl S tagged chiral from 3D (Astex 1hnn)
    ],
)
def test_the_rmsd_equals_rdkits_calcrms_on_the_heavy_atoms(smiles, tmp_path):
    mol = _from_sdf(smiles, tmp_path)
    poses = _poses(mol, 6)

    scores = RMSD(mol).batch_scores(poses)
    matrix = rmsd_matrix(mol, poses)

    for i in range(6):
        posed_i = mol.to_rdkit(poses[i])
        assert scores[i] == pytest.approx(_heavy_rms(posed_i, mol.rdkit), abs=1e-6)
        for j in range(i + 1, 6):
            assert matrix[i, j] == pytest.approx(
                _heavy_rms(posed_i, mol.to_rdkit(poses[j])), abs=1e-6
            )


def test_swapping_the_two_oxygens_of_a_carboxylic_acid_gives_zero(tmp_path):
    mol = _from_sdf("OC(=O)Cc1ccccc1", tmp_path)
    heavy = Chem.RemoveAllHs(mol.rdkit, sanitize=False)
    oxygens = [a.GetIdx() for a in heavy.GetAtoms() if a.GetSymbol() == "O"]
    ref = Mol(heavy, flexible=True, hydrogens="keep")
    conformer, positions = ref.rdkit.GetConformer(), ref.get_positions()
    a, b = oxygens
    conformer.SetAtomPosition(a, tuple(float(x) for x in positions[b]))
    conformer.SetAtomPosition(b, tuple(float(x) for x in positions[a]))
    probe = Mol(heavy, flexible=True, hydrogens="keep")

    assert RMSD(probe, ref).get_score(probe.input_pose) < 1e-9
