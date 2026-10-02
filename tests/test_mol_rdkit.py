"""`Mol` holds an RDKit molecule (not is one): the explicit RDKit access, and copying/pickling."""

import copy
import pickle

import numpy as np
import pytest
from rdkit import Chem

from pyrite import Mol
from pyrite.scoring import ElectroStatic


@pytest.fixture
def ligand(ctx) -> Mol:
    """A fresh copy, as these tests modify the molecule."""
    return ctx.ligand.copy()


def test_a_mol_is_not_an_rdkit_molecule(ligand):
    assert not isinstance(ligand, Chem.Mol)
    assert isinstance(ligand.rdkit, Chem.Mol)
    assert ligand.rdkit is ligand.rdkit  # the held molecule, not a new copy on every access


def test_the_explicit_accessors_match_the_rdkit_molecule(ligand):
    assert ligand.n_atoms == ligand.rdkit.GetNumAtoms()
    assert [a.GetIdx() for a in ligand.atoms] == list(range(ligand.n_atoms))
    assert ligand.n_conformers == ligand.rdkit.GetNumConformers() == 1

    conf_id = ligand.pose_to_conformer(np.zeros(ligand.layout.n_dims), new_conf=True)
    assert ligand.n_conformers == 2
    ligand.remove_conformer(conf_id)
    assert ligand.n_conformers == 1


def test_descriptors_and_other_rdkit_functions_work_on_the_held_molecule(ligand):
    from rdkit.Chem import Descriptors, rdMolDescriptors

    assert Descriptors.MolWt(ligand.rdkit) > 100
    assert rdMolDescriptors.CalcNumRotatableBonds(ligand.rdkit) >= ligand.n_tors


def test_the_molecule_passed_in_is_never_modified():
    source = Chem.MolFromSmiles("CCOc1ccccc1")
    n_atoms, charge = source.GetNumAtoms(), source.GetAtomWithIdx(0).GetFormalCharge()

    mol = Mol(source, flexible=True)
    mol.rdkit.GetAtomWithIdx(0).SetFormalCharge(1)

    assert source.GetNumAtoms() == n_atoms
    assert source.GetNumConformers() == 0  # Pyrite embedded its own conformer
    assert source.GetAtomWithIdx(0).GetFormalCharge() == charge


def test_to_rdkit_is_an_independent_copy_that_can_be_edited(ligand):
    editable = Chem.RWMol(ligand.to_rdkit())
    editable.AddAtom(Chem.Atom(6))

    assert editable.GetNumAtoms() == ligand.n_atoms + 1
    assert ligand.n_atoms == ligand.rdkit.GetNumAtoms()
    assert ligand.to_rdkit() is not ligand.rdkit
    # ... and a new Mol can be built from the edited molecule
    assert Mol(editable.GetMol(), flexible=False).n_atoms == ligand.n_atoms + 1


def test_a_mol_cannot_be_built_from_a_mol(ligand):
    with pytest.raises(TypeError, match="rdkit"):
        Mol(ligand)


def test_a_copy_is_complete_and_independent(ctx, ligand):
    ligand.pose_to_conformer(ctx.poses[0])  # move the global conformer first
    clone = ligand.copy()

    assert type(clone) is Mol and clone.rdkit is not ligand.rdkit
    assert clone.layout == ligand.layout and clone.center_atom == ligand.center_atom
    assert np.array_equal(clone.atom_types, ligand.atom_types)
    assert np.array_equal(clone.scoring_mask, ligand.scoring_mask)
    assert list(map(tuple, clone.rotatable_torsions)) == list(map(tuple, ligand.rotatable_torsions))
    assert np.array_equal(clone._reference_positions, ligand._reference_positions)
    assert np.array_equal(clone.get_positions(), ligand.get_positions())  # the moved state is kept
    assert np.array_equal(clone.pose_to_positions(ctx.poses), ligand.pose_to_positions(ctx.poses))

    clone.pose_to_conformer(ctx.poses[1])
    clone.rdkit.GetAtomWithIdx(0).SetProp("changed", "yes")
    assert not np.allclose(clone.get_positions(), ligand.get_positions())
    assert not ligand.rdkit.GetAtomWithIdx(0).HasProp("changed")


@pytest.mark.parametrize("copier", [copy.copy, copy.deepcopy])
def test_the_copy_module_gives_a_full_independent_copy(ctx, ligand, copier):
    clone = copier(ligand)

    assert type(clone) is Mol and clone.rdkit is not ligand.rdkit
    assert clone.n_tors == ligand.n_tors and clone.layout == ligand.layout


def test_a_pickled_mol_keeps_everything_scoring_needs(ctx, ligand):
    ligand.rdkit.GetAtomWithIdx(2).SetProp("custom", "yes")
    default = Chem.GetDefaultPickleProperties()

    clone = pickle.loads(pickle.dumps(ligand))

    assert Chem.GetDefaultPickleProperties() == default  # RDKit's global setting is restored
    assert type(clone) is Mol and clone.n_tors == ligand.n_tors
    assert np.array_equal(clone.get_positions(), ligand.get_positions())  # not truncated to float32
    assert np.array_equal(clone.atom_types, ligand.atom_types)
    assert clone.rdkit.GetAtomWithIdx(2).GetProp("custom") == "yes"
    charges = lambda m: [a.GetDoubleProp("_GasteigerCharge") for a in m.atoms]  # noqa: E731
    assert np.array_equal(charges(clone), charges(ligand))
    # a scoring function that reads the charges gives the same score for the unpickled molecule
    pose = ctx.poses[0]
    assert ElectroStatic(clone, ctx.receptor).get_score(pose) == pytest.approx(
        ElectroStatic(ligand, ctx.receptor).get_score(pose), rel=1e-12
    )


def test_molecules_are_hashable_by_identity(ligand):
    other = ligand.copy()

    assert {ligand: 1, other: 2}[ligand] == 1
    assert ligand != other and ligand == ligand  # noqa: PLR0124
