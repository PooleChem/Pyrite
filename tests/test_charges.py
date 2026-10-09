"""Formal charges: taken from the input where it states them, derived only where it cannot."""

import numpy as np
import pytest
from conftest import EXAMPLES
from rdkit import Chem
from rdkit.Chem import AllChem

from pyrite import Mol

pytestmark = pytest.mark.filterwarnings("ignore:from_pdb")


def _charges(mol: Mol) -> list[int]:
    return [a.GetFormalCharge() for a in mol.atoms]


@pytest.mark.parametrize("smiles", ["CCO", "CC(=O)O", "OCC(O)CO", "Oc1ccccc1"])
def test_a_hydroxyl_in_a_smiles_string_stays_neutral(smiles):
    # "O" with one bond in a SMILES string is an OH: it used to become O- (ethanol an ethoxide).
    assert set(_charges(Mol.from_smiles(smiles))) == {0}


@pytest.mark.parametrize("smiles, n_hydrogens", [("CCO", 6), ("CC(=O)O", 4), ("Oc1ccccc1", 6)])
def test_adding_hydrogens_completes_the_hydroxyls(smiles, n_hydrogens):
    mol = Mol.from_smiles(smiles, hydrogens="add")

    assert sum(a.GetAtomicNum() == 1 for a in mol.atoms) == n_hydrogens


@pytest.mark.parametrize(
    "smiles, charge", [("CC(=O)[O-]", -1), ("C[N+](C)(C)C", 1), ("c1cc[n+]([O-])cc1", 0)]
)
def test_charges_stated_in_the_input_are_kept(smiles, charge):
    assert sum(_charges(Mol.from_smiles(smiles))) == charge


def test_a_nitrogen_with_four_bonds_drawn_neutral_is_an_ammonium():
    rd = Chem.AddHs(Chem.MolFromSmiles("C[N+](C)(C)C"))
    AllChem.EmbedMolecule(rd, randomSeed=1)
    nitrogen = next(a for a in rd.GetAtoms() if a.GetSymbol() == "N")
    nitrogen.SetFormalCharge(0)  # as files without charges have it

    mol = Mol(rd, hydrogens="keep")

    assert mol.rdkit.GetAtomWithIdx(nitrogen.GetIdx()).GetFormalCharge() == 1


def test_a_heavy_atom_sdf_with_a_hydroxyl_stays_neutral(tmp_path):
    rd = Chem.AddHs(Chem.MolFromSmiles("OCc1ccc(C(=O)O)cc1"))
    AllChem.EmbedMolecule(rd, randomSeed=1)
    path = str(tmp_path / "heavy.sdf")
    with Chem.SDWriter(path) as writer:
        writer.write(Chem.RemoveHs(rd))

    assert set(_charges(Mol.from_sdf(path))) == {0}


def _by_name(mol: Mol) -> dict:
    """The formal charges by (residue name, atom name)."""
    out = {}
    for atom in mol.atoms:
        info = atom.GetPDBResidueInfo()
        out.setdefault((info.GetResidueName(), info.GetName().strip()), set()).add(
            atom.GetFormalCharge()
        )
    return out


@pytest.mark.parametrize("hydrogens", ["polar", "keep", "remove"])
def test_a_pdb_file_with_hydrogens_gets_its_ionised_groups(hydrogens):
    # factor_x.pdb has every hydrogen: an oxygen with one bond and no H has lost its proton.
    charges = _by_name(Mol.from_pdb(str(EXAMPLES / "factor_x.pdb"), hydrogens=hydrogens))

    assert charges[("ASP", "OD2")] == {-1} and charges[("GLU", "OE2")] == {-1}
    assert charges[("LYS", "NZ")] == {1}
    assert charges[("SER", "OG")] == {0} and charges[("TYR", "OH")] == {0}


def test_a_pdb_file_without_hydrogens_gets_no_oxides(tmp_path):
    # Without hydrogens a hydroxyl and a carboxylate oxygen look the same: nothing is charged.
    lines = (EXAMPLES / "factor_x.pdb").read_text().splitlines()
    heavy = [line for line in lines if not (line[:4] == "ATOM" and line[76:78].strip() == "H")]
    path = tmp_path / "heavy.pdb"
    path.write_text("\n".join(heavy) + "\n")

    charges = _by_name(Mol.from_pdb(str(path)))

    assert charges[("SER", "OG")] == {0} and charges[("TYR", "OH")] == {0}
    assert charges[("ASP", "OD2")] == {0}
    assert np.all(np.array(_charges(Mol.from_pdb(str(path)))) >= 0)


def test_the_oxide_rule_reads_the_valences_after_atoms_were_removed():
    # from_pdb removes residues with impossible valences before charging; RDKit keeps the old
    # valences cached. An oxygen that lost its bonded partner must be read with its new valence
    # (Astex 1mzc: a C-terminal OXT whose neighbouring residue was removed).
    from pyrite._common import _charge_deprotonated_oxygens

    rd = Chem.AddHs(Chem.MolFromSmiles("CC(=O)OC"))
    rd.UpdatePropertyCache(strict=False)
    methyl = 4  # the O-methyl carbon of the ester
    oxygen = 3
    edit = Chem.RWMol(rd)
    for idx in sorted(
        [
            methyl,
            *(
                n.GetIdx()
                for n in rd.GetAtomWithIdx(methyl).GetNeighbors()
                if n.GetAtomicNum() == 1
            ),
        ],
        reverse=True,
    ):
        edit.RemoveAtom(idx)
    mol = edit.GetMol()  # the ester oxygen now has one bond, but its cached valence is 2

    _charge_deprotonated_oxygens(mol)

    assert mol.GetAtomWithIdx(oxygen).GetFormalCharge() == -1
