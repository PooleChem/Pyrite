---
file_format: mystnb
kernelspec:
  name: python3
---

# Molecules

Everything in Pyrite starts with a {class}`~pyrite.Mol`: the ligand that is docked, and the receptor
it is docked into. A `Mol` holds an RDKit molecule, and adds what docking needs: atom types, the
rotatable bonds, and the poses of the molecule (see the next page).

## Loading

A ligand is usually loaded from an SDF file, which has its bond orders. This one has no hydrogens,
so they are added (see [Hydrogens](#hydrogens)):

```{code-cell} python
from pyrite import Mol

ligand = Mol.from_sdf("input/2boh_ligand.sdf", flexible=True, hydrogens="add")
ligand.n_atoms, ligand.n_tors
```

or made from a SMILES string, which gets a 3D conformer from RDKit:

```{code-cell} python
paracetamol = Mol.from_smiles("CC(=O)Nc1ccc(O)cc1", hydrogens="add", flexible=True)
paracetamol.n_atoms, paracetamol.n_tors
```

A receptor is loaded from a PDB file. A file straight from the Protein Data Bank also holds waters,
ions, ligands and sometimes non-standard residues; {func}`~pyrite.io.fix_receptor_pdb` leaves only
the protein:

```{code-cell} python
from pyrite.io import fix_receptor_pdb

receptor = Mol.from_pdb(fix_receptor_pdb("input/2boh.pdb"), hydrogens="add")
receptor.n_atoms
```

A molecule already in RDKit is loaded with {meth}`~pyrite.Mol.from_rdkit`. Every `from_*` method
passes its keyword arguments on to {class}`~pyrite.Mol`, such as `flexible` and `hydrogens`.

## Hydrogens

Hydrogens matter for one thing in particular: the polar hydrogens (on N, O and S) decide which
atoms can donate a hydrogen bond. `hydrogens` decides which hydrogens a molecule has:

| `hydrogens` | |
|---|---|
| `"polar"` (default) | keep the polar hydrogens in the file, remove the others |
| `"add"` | add the missing hydrogens |
| `"keep"` | keep the hydrogens as they are in the file |
| `"remove"` | remove all hydrogens |

`"polar"` adds no hydrogens. A file without hydrogens, such as a PDB file from the Protein Data Bank,
then has no hydrogen bond donors at all:

```{code-cell} python
from pyrite import AtomType


def donors(mol):
    return sum("Donor" in AtomType(t).name for t in mol.atom_types)


fixed = fix_receptor_pdb("input/2boh.pdb")
donors(Mol.from_pdb(fixed)), donors(Mol.from_pdb(fixed, hydrogens="add"))
```

So a file without hydrogens is loaded with `hydrogens="add"`. RDKit adds them to neutral groups:
it cannot know a protonation state that the file does not give.

Hydrogens are left out of scoring by default (`ignore_hydrogens=True`): scoring functions use the
atoms in {attr}`~pyrite.Mol.scoring_mask`, the heavy atoms.

## Atom types

Every atom gets an {class}`~pyrite.AtomType`, which the scoring functions use: its element, and
whether it is a hydrogen bond donor or acceptor, or hydrophobic.

```{code-cell} python
from collections import Counter

Counter(AtomType(t).name for t in ligand.atom_types)
```

## Flexibility

A `flexible` molecule can change its shape: every rotatable bond is a torsion, which a search can
turn. A molecule that is not flexible is rigid: it can only be rotated and moved.

```{code-cell} python
ligand.n_tors, ligand.torsions
```

{attr}`~pyrite.Mol.torsions` are the current angles of the torsions, in radians, and
{attr}`~pyrite.Mol.rotatable_torsions` the four atoms of each. Bonds to a terminal group, such as a
methyl or hydroxyl group, only turn hydrogens, and are not rotatable unless `flex_hydrogens=True`.

The center atom is the point the molecule rotates around, and the side of every torsion that stays
in place. It is chosen so that no torsion moves a large part of the molecule:

```{code-cell} python
ligand.set_draw_options({"size": (600, 350), "highlight": "center"})
ligand.svg
```

## Positions

A molecule has the positions of its atoms, and the pose those positions are in, its
{attr}`~pyrite.Mol.input_pose`. For a ligand from a crystal structure, that is the crystal pose.

```{code-cell} python
ligand.positions[:3]
```

Poses, and how a pose becomes positions, are the subject of the next page.

## Viewing

A molecule at the end of a cell is shown in 3D; {class}`~pyrite.Viewer` shows several objects
together, each with its own draw options:

```{code-cell} python
from pyrite import Viewer

viewer = Viewer(width=600, height=400)
viewer.add(receptor, options={"surfaceopacity": 0.4})
viewer.add(ligand).zoom_to(ligand)
```

## RDKit

{attr}`~pyrite.Mol.rdkit` is the RDKit molecule, for any RDKit function that reads a molecule. It is
read-only: changing it would leave the atom types and torsions of the `Mol` behind.
{meth}`~pyrite.Mol.to_rdkit` gives a copy to change, and a new `Mol` is built from it:

```{code-cell} python
from rdkit.Chem import Descriptors

round(Descriptors.MolWt(ligand.rdkit), 2)
```

## Saving

{meth}`~pyrite.Mol.to_sdf` writes a molecule to an SDF file, in its current positions or in poses:

```{code-cell} python
ligand.to_sdf("ligand_copy.sdf")
```

```{code-cell} python
:tags: [remove-cell]

import os

os.remove("ligand_copy.sdf")
```
