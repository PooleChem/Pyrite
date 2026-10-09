"""The default center atom: balanced, from the molecular graph, and the fixed side of every torsion."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from pyrite import Mol

BRANCHED = "CCCC(CCO)(CC(=O)OC)c1ccccc1OCC"
CHAIN = "CCCCCCCCOC(=O)CCNC(=O)c1ccccc1"


def _embedded(smiles: str, seed: int = 1) -> Chem.Mol:
    rd = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(rd, randomSeed=seed)
    return Chem.RemoveHs(rd)


def _heavy(mol: Mol) -> list[int]:
    return [a.GetIdx() for a in mol.atoms if a.GetAtomicNum() > 1]


def _largest_moved(mol: Mol) -> int:
    heavy = set(_heavy(mol))
    return max((len(heavy & set(m.tolist())) for m in mol._Mol__torsion_moving), default=0)


@pytest.fixture(scope="module")
def molecules(ctx) -> dict[str, Mol]:
    return {
        "factor_x": ctx.ligand,
        "branched": Mol(_embedded(BRANCHED), flexible=True),
        "chain": Mol(_embedded(CHAIN), flexible=True),
        "biphenyl": Mol(_embedded("c1ccccc1-c1ccccc1"), flexible=True),
    }


@pytest.mark.parametrize("name", ["factor_x", "branched", "chain", "biphenyl"])
def test_no_other_center_atom_moves_fewer_atoms_with_its_largest_torsion(molecules, name):
    mol = molecules[name]

    best = min(_largest_moved(Mol(mol.rdkit, flexible=True, center_atom=i)) for i in _heavy(mol))

    assert _largest_moved(mol) == best


def test_the_balanced_center_beats_the_centroid_choice_where_they_differ(molecules):
    # factor_x: the heavy atom closest to the centroid made one torsion move 21 of 33 atoms.
    mol = molecules["factor_x"]
    positions = mol.get_positions()
    heavy = _heavy(mol)
    centroid_atom = heavy[
        int(np.argmin(np.linalg.norm(positions[heavy] - positions[heavy].mean(axis=0), axis=1)))
    ]

    assert mol.center_atom != centroid_atom
    assert _largest_moved(mol) < _largest_moved(
        Mol(mol.rdkit, flexible=True, center_atom=centroid_atom)
    )


def test_the_center_atom_is_never_moved_by_a_torsion(molecules):
    for mol in molecules.values():
        assert all(mol.center_atom not in m.tolist() for m in mol._Mol__torsion_moving)


@pytest.mark.parametrize(
    "smiles",
    [
        BRANCHED,
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
        "CN1CCC(CC1)Oc1ccc(cc1)C(=O)NCCc1ccccc1",
    ],
)
def test_the_center_atom_does_not_depend_on_the_loaded_conformer(smiles):
    # Breaking ties by the geometry before the topology gives a different atom for different
    # conformers of these molecules (ibuprofen: atoms 6 and 8, instead of always atom 7).
    centers = {Mol(_embedded(smiles, seed), flexible=True).center_atom for seed in range(20)}

    assert len(centers) == 1


def test_a_given_center_atom_is_used_and_does_not_change_the_torsions(molecules):
    mol = molecules["branched"]
    other = Mol(mol.rdkit, flexible=True, center_atom=_heavy(mol)[-1])

    assert other.center_atom == _heavy(mol)[-1] != mol.center_atom
    # the same rotatable bonds with the same angles, only their direction can differ
    bonds = lambda m: [frozenset(t[1:3]) for t in m.rotatable_torsions]  # noqa: E731
    assert bonds(mol) == bonds(other)
    assert np.allclose(
        np.angle(np.exp(1j * (np.array(mol.torsions) - np.array(other.torsions)))), 0, atol=1e-9
    )


def test_without_torsions_it_is_the_heavy_atom_closest_to_the_centroid():
    for flexible in (False, True):
        mol = Mol(_embedded("c1ccccc1O"), flexible=flexible)  # no rotatable bond
        positions = mol.get_positions()
        heavy = _heavy(mol)
        expected = heavy[
            int(np.argmin(np.linalg.norm(positions[heavy] - positions[heavy].mean(axis=0), axis=1)))
        ]

        assert mol.center_atom == expected


def test_a_pose_translation_is_where_the_center_atom_ends_up(molecules):
    for mol in molecules.values():
        layout, rng = mol.layout, np.random.default_rng(0)
        rotation = layout.sample_random_rotations(6, rng)
        translation = rng.uniform(-5, 5, (6, 3))
        torsions = rng.uniform(-np.pi, np.pi, (6, layout.n_tors))
        poses = np.concatenate([rotation, translation, torsions], axis=1)

        positions = mol.pose_to_positions(poses)

        assert np.allclose(positions[:, mol.center_atom], translation, atol=1e-9)
