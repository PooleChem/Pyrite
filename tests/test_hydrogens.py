"""What `hydrogens='keep' | 'add' | 'remove'` does, in the constructor and in every `from_*`."""

import numpy as np
import pytest
from conftest import EXAMPLES
from rdkit import Chem
from rdkit.Chem import AllChem

from pyrite import Mol

LIGAND = str(EXAMPLES / "factor_x_ligand.sdf")


def _n_hydrogens(mol: Mol) -> int:
    return sum(a.GetAtomicNum() == 1 for a in mol.atoms)


def _with_hydrogens(smiles: str = "CCO") -> Chem.Mol:
    rd = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(rd, randomSeed=1)
    return rd


def _hydrogen_bonds(mol: Mol) -> np.ndarray:
    """The lengths of the X-H bonds."""
    positions = mol.get_positions()
    return np.array(
        [
            np.linalg.norm(positions[b.GetBeginAtomIdx()] - positions[b.GetEndAtomIdx()])
            for b in mol.rdkit.GetBonds()
            if 1 in (b.GetBeginAtom().GetAtomicNum(), b.GetEndAtom().GetAtomicNum())
        ]
    )


def test_keep_leaves_the_hydrogens_as_they_are():
    with_hs, without_hs = _with_hydrogens(), Chem.RemoveHs(_with_hydrogens())

    assert _n_hydrogens(Mol(with_hs, hydrogens="keep")) == 6
    assert _n_hydrogens(Mol(without_hs, hydrogens="keep")) == 0


def test_remove_removes_all_hydrogens_and_keeps_the_other_coordinates():
    rd = _with_hydrogens("CC(=O)Nc1ccc(O)cc1")
    heavy = [a.GetIdx() for a in rd.GetAtoms() if a.GetAtomicNum() > 1]

    mol = Mol(rd, hydrogens="remove")

    assert _n_hydrogens(mol) == 0 and mol.n_atoms == len(heavy)
    assert np.allclose(mol.get_positions(), rd.GetConformer().GetPositions()[heavy])


def test_add_adds_the_missing_hydrogens_with_coordinates_and_keeps_the_others():
    heavy_only = Mol.from_sdf(LIGAND, hydrogens="keep")
    plain = Mol.from_sdf(LIGAND, hydrogens="keep").get_positions()

    mol = Mol.from_sdf(LIGAND, hydrogens="add")

    assert heavy_only.n_atoms == 33 and _n_hydrogens(heavy_only) == 0
    assert _n_hydrogens(mol) == 27 and mol.n_atoms == 60
    assert np.allclose(mol.get_positions()[:33], plain)  # the existing atoms did not move
    assert np.all((_hydrogen_bonds(mol) > 0.9) & (_hydrogen_bonds(mol) < 1.2))  # real positions


def test_add_to_a_molecule_that_has_hydrogens_adds_nothing():
    rd = _with_hydrogens()

    assert Mol(rd, hydrogens="add").n_atoms == rd.GetNumAtoms()


@pytest.mark.parametrize("hydrogens, expected", [("keep", 0), ("remove", 0), ("add", 5)])
def test_from_smiles(hydrogens, expected):
    mol = Mol.from_smiles("CCO", hydrogens=hydrogens)

    assert _n_hydrogens(mol) == expected
    assert mol.n_conformers == 1 and mol.rdkit.GetConformer().Is3D()
    assert mol.n_atoms == 3 + expected  # the temporary hydrogens of the embedding are not kept


@pytest.mark.filterwarnings("ignore:from_pdb")
def test_the_receptor_has_no_hydrogens_at_the_origin_when_they_are_added():
    # from_pdb used to add hydrogens without coordinates: 41 of them sat at (0, 0, 0)
    receptor = Mol.from_pdb(str(EXAMPLES / "factor_x.pdb"), hydrogens="add")
    hydrogens = [a.GetIdx() for a in receptor.atoms if a.GetAtomicNum() == 1]

    assert np.all(np.linalg.norm(receptor.get_positions()[hydrogens], axis=1) > 1e-6)
    assert _n_hydrogens(receptor) > _n_hydrogens(
        Mol.from_pdb(str(EXAMPLES / "factor_x.pdb"), hydrogens="keep")
    )


def test_from_pdb_remove_leaves_heavy_atoms_only(ctx):
    assert _n_hydrogens(ctx.receptor) == 0  # the shared receptor is loaded with the default


def test_an_unknown_value_is_rejected():
    with pytest.raises(ValueError, match="hydrogens"):
        Mol(_with_hydrogens(), hydrogens="some")


def test_the_molecule_passed_in_keeps_its_hydrogens():
    rd = _with_hydrogens()

    Mol(rd, hydrogens="remove")

    assert rd.GetNumAtoms() == 9


def test_embedding_uses_temporary_hydrogens_for_a_better_geometry():
    # Embedding the heavy atoms alone gives a more strained geometry: 443 against 79 kcal/mol
    # above the relaxed structure for this molecule.
    smiles = "COc1ccc(CCN(C)CCCC(C#N)(C(C)C)c2ccc(OC)c(OC)c2)cc1OC"

    def strain(heavy_atoms: Chem.Mol) -> float:
        with_hs = Chem.AddHs(heavy_atoms, addCoords=True)
        field = AllChem.MMFFGetMoleculeForceField(
            with_hs, AllChem.MMFFGetMoleculeProperties(with_hs)
        )
        energy = field.CalcEnergy()
        field.Minimize(maxIts=2000)
        return energy - field.CalcEnergy()

    heavy_only = Chem.MolFromSmiles(smiles)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    AllChem.EmbedMolecule(heavy_only, params)

    assert strain(Mol.from_smiles(smiles).rdkit) < 0.5 * strain(heavy_only)
