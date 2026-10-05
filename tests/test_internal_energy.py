"""`InternalEnergy` is the MMFF energy of the complete molecule, whichever hydrogens the Mol kept."""

import numpy as np
import pytest
from helpers import loaded_pose
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from pyrite import Mol, Poses
from pyrite.scoring import InternalEnergy

SMILES = "CC(=O)Nc1ccc(OCC(C)O)cc1"  # methyls (C-H), an amide N-H and an O-H


def _complete_molecule() -> Chem.Mol:
    """Heavy atoms from an embedding, every hydrogen where RDKit's AddHs places it."""
    rd = Chem.AddHs(Chem.MolFromSmiles(SMILES))
    AllChem.EmbedMolecule(rd, randomSeed=4)
    return Chem.AddHs(Chem.RemoveHs(rd), addCoords=True)


def _poses(mol: Mol, n: int = 20) -> Poses:
    rng = np.random.default_rng(0)
    values = np.tile(np.array(loaded_pose(mol)), (n, 1))
    values[:, mol.layout.tors_slice] += rng.normal(0, 1.0, (n, mol.layout.n_tors))
    return Poses(values, mol.layout)


def _mmff(rd: Chem.Mol) -> float:
    return AllChem.MMFFGetMoleculeForceField(rd, AllChem.MMFFGetMoleculeProperties(rd)).CalcEnergy()


def test_a_molecule_with_all_its_hydrogens_is_scored_as_it_is():
    mol = Mol(_complete_molecule(), flexible=True, hydrogens="keep")
    poses = _poses(mol)

    expected = [_mmff(mol.to_rdkit(p)) for p in poses]

    assert InternalEnergy(mol).batch_scores(poses) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("hydrogens", ["polar", "remove"])
def test_missing_hydrogens_are_added_to_every_pose(hydrogens):
    mol = Mol(_complete_molecule(), flexible=True, hydrogens=hydrogens)
    poses = _poses(mol)

    # Independent path: complete each posed molecule with AddHs and score it with a fresh MMFF.
    expected = []
    for pose in poses:
        posed = mol.to_rdkit(pose)
        posed.UpdatePropertyCache(strict=False)
        expected.append(_mmff(Chem.AddHs(posed, addCoords=True)))

    assert InternalEnergy(mol).batch_scores(poses) == pytest.approx(expected, rel=1e-9)


def test_all_hydrogen_settings_agree_when_the_hydrogens_are_where_rdkit_puts_them():
    complete = _complete_molecule()
    energies = {}
    for hydrogens in ("keep", "polar", "remove"):
        mol = Mol(complete, flexible=True, hydrogens=hydrogens)
        energies[hydrogens] = InternalEnergy(mol).get_score(loaded_pose(mol))

    assert energies["polar"] == pytest.approx(energies["keep"], rel=1e-9)
    assert energies["remove"] == pytest.approx(energies["keep"], rel=1e-9)


def test_mmff_no_longer_scores_a_molecule_without_its_hydrogens(capfd):
    RDLogger.EnableLog("rdApp.warning")
    mol = Mol(_complete_molecule(), flexible=True, hydrogens="polar")

    InternalEnergy(mol).get_score(loaded_pose(mol))

    assert "does not have explicit Hs" not in capfd.readouterr().err
