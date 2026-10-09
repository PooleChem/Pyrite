"""Which atoms count is the molecule's `scoring_mask`: no scoring function has a switch of its own."""

import numpy as np
import pytest
from conftest import EXAMPLES
from rdkit import Chem
from rdkit.Chem import AllChem

from pyrite import Mol
from pyrite.scoring import (
    DistanceToPocket,
    InternalOverlap,
    NumAtoms,
    NumProteinAtomsWithinA,
    WeightedBoundsOverlap,
)


def _molecule(**kwargs) -> Mol:
    rd = Chem.AddHs(Chem.MolFromSmiles("CCCC(CCO)(CC(=O)OC)c1ccccc1OCC"))
    AllChem.EmbedMolecule(rd, randomSeed=1)
    return Mol(rd, hydrogens="keep", flexible=True, **kwargs)


def test_the_flag_of_the_molecule_excludes_every_hydrogen_or_none():
    default, counting = _molecule(), _molecule(ignore_hydrogens=False)
    hydrogens = np.array([a.GetAtomicNum() == 1 for a in default.atoms])

    assert hydrogens.sum() > 10
    assert np.array_equal(default.scoring_mask, ~hydrogens)  # polar and non-polar alike
    assert counting.scoring_mask.all()
    with pytest.raises(TypeError):
        Mol(Chem.MolFromSmiles("CCO"), ignore_non_polar_hydrogens=True)  # the old name


def test_the_per_term_switches_are_gone(ctx):
    mol = ctx.ligand
    for build in (
        lambda: InternalOverlap(mol, ignore_hs=True),
        lambda: NumAtoms(mol, include_hs=True),
        lambda: NumProteinAtomsWithinA(mol, ctx.receptor, ignore_hs_ligand=True),
        lambda: DistanceToPocket(mol, ctx.pocket, include_hs=True),
        lambda: WeightedBoundsOverlap(mol, ctx.pocket, include_hs=True),
    ):
        with pytest.raises(TypeError):
            build()


@pytest.mark.parametrize("ignore", [True, False])
def test_every_term_takes_its_atoms_from_the_scoring_mask(ctx, ignore):
    mol = _molecule(ignore_hydrogens=ignore)
    mask = np.asarray(mol.scoring_mask)

    n_hydrogens = sum(a.GetAtomicNum() == 1 for a in mol.atoms)
    assert (
        NumAtoms(mol).result == mask.sum() == (mol.n_atoms - n_hydrogens if ignore else mol.n_atoms)
    )
    assert np.array_equal(DistanceToPocket(mol, ctx.pocket)._mask, mask)
    assert np.array_equal(WeightedBoundsOverlap(mol, ctx.pocket).mask, mask)
    assert np.array_equal(NumProteinAtomsWithinA(mol, ctx.receptor)._mask, mask)
    overlap = InternalOverlap(mol)
    assert mask[overlap._first].all() and mask[overlap._second].all()


@pytest.mark.filterwarnings("ignore:from_pdb")
def test_the_protein_atom_count_does_not_depend_on_the_hydrogens_in_the_file(ctx):
    # The receptor side used to count every atom, so the number depended on whether the file
    # (or `hydrogens=`) kept hydrogens.
    path = str(EXAMPLES / "factor_x.pdb")
    counts = {}
    for hydrogens in ("keep", "polar", "remove"):
        receptor = Mol.from_pdb(path, hydrogens=hydrogens)
        term = NumProteinAtomsWithinA(ctx.ligand, receptor, a=4.0)
        counts[hydrogens] = [term.get_score(p) for p in ctx.poses]

    assert counts["keep"] == counts["polar"] == counts["remove"]
    assert max(counts["keep"]) > 0
