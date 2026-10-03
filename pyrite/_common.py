from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import py3Dmol
from IPython.display import SVG, Image, display
from numpy.typing import NDArray
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, SDWriter
from scipy.spatial.transform import Rotation

from ._util import (
    _apply_torsions,
    _atoms_beyond,
    _pack_torsions,
)
from .atom_consts import AtomType, vina_atom_consts
from .view import Viewer

ROTATABLE_BOND_STRUCT = Chem.MolFromSmarts(
    "[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])"
    "([CH3])[CH3])&!$([CD3](=[N,O,S])-!@[#7,O,S!D1])&!$([#7,O,S!D1]-!@[CD3]"
    "=[N,O,S])&!$([CD3](=[N+])-!@[#7!D1])&!$([#7!D1]-!@[CD3]=[N+])]-,:;!@"
    "[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])"
    "([CH3])[CH3])]"
)

HBA_STRUCT = Chem.MolFromSmarts(
    "[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0])]"
)

NON_POLAR_H_STRUCT = Chem.MolFromSmarts("[#1;$([#1]-[#6,#14])]")

VIEWER_PROTEINS_HEAVY_ATOMS_CUTOFF = 1000

# RDKit stops looking for substructure matches after 1000 unless told otherwise, which silently
# truncates the matches of a query on a large molecule such as a receptor.
_MAX_MATCHES = 10_000_000


class Mol:
    """
    Representation of a molecule.

    The Mol class provides methods for initializing molecules from various sources,
    such as SMILES strings, PDB files, or objects. It also assigns atom
    types, rotatable torsions, and the center point of the molecule. Furthermore, it allows for
    easy manipulation of ligand position, rotation, and torsion angles.

    A ``Mol`` *holds* an RDKit molecule (it is not one): use :attr:`rdkit` for any RDKit
    functionality (descriptors, substructure matches, force fields), treating it as read-only,
    and :meth:`to_rdkit` for a copy to edit. A new topology is a new ``Mol``: ``Mol(edited)``.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The :class:`~rdkit.Chem.rdchem.Mol` object representing the molecule.
    center_atom : int, optional
        The index of the atom to use as the center point for rotations, and the fixed side of
        every torsion. By default, for a ``flexible`` molecule it is chosen from the molecular
        graph so that no torsion moves a large part of the molecule (the largest moved fragment is
        minimal, ties are broken by centrality), independent of the loaded conformer. Without
        torsions it is the heavy atom closest to the centroid of the loaded conformer.
    hydrogens : {'keep', 'add', 'remove', 'polar'}, default 'polar'
        What to do with the explicit hydrogens of `mol`: keep them as they are, add the missing
        ones (with coordinates), remove all of them, or keep only the polar ones (those on
        N, O, S, ...; the hydrogens on carbon are removed).
    flex_hydrogens : bool, default False
        Whether bonds to a terminal heavy atom (a methyl group, a hydroxyl hydrogen) are
        rotatable torsions too.
    flexible : bool, default False
        Whether the rotatable bonds of the molecule are torsions of its poses.
    rotation_type : {'euler', 'quat'}, default 'euler'
        How the rotation of a pose is represented.
    ignore_hydrogens : bool, default True
        Whether hydrogen atoms (polar and non-polar alike) are excluded from `scoring_mask`, the
        mask every scoring function built on this molecule uses to decide which atoms count.
        Polar hydrogens still decide the atom types of their neighbours (donors); only the
        hydrogens themselves are left out of the scores, as in united-atom scoring.


    """

    # region Construction

    def __init__(
        self,
        mol: Chem.Mol = None,
        hydrogens: Literal["keep", "add", "remove", "polar"] = "polar",
        flex_hydrogens: bool = False,
        flexible: bool = False,  # TODO: allow list of resids. No. Read everything rigid, and allow for auto setting of torsions, or manual, or by resid.
        center_atom: int = None,
        rotation_type: Literal["euler", "quat"] = "euler",
        ignore_hydrogens: bool = True,
    ):
        if isinstance(mol, Mol):
            raise TypeError(
                "A Mol is built from an RDKit molecule: use `Mol(mol.rdkit)` to build it again, "
                "or `mol.copy()` for an exact copy."
            )
        self._rdkit = Chem.Mol(mol)  # a private copy: the caller's molecule is never modified
        self.__rotatable_torsions = np.array([], dtype=object)
        self.__torsion_moving = []
        self.__packed_torsions = _pack_torsions([], [])

        self._fix_mol_valence(sanitize=False)  # TODO: sanitize?

        if hydrogens == "add":
            self._rdkit = Chem.AddHs(self._rdkit, addCoords=True)
        elif hydrogens == "remove":
            self._rdkit = Chem.RemoveHs(self._rdkit, sanitize=False)
        elif hydrogens == "polar":
            # united atom: the hydrogens on carbon (and silicon) are merged into their atom, the
            # polar ones (on N, O, S, ...) stay, as they decide the donor atom types
            editable = Chem.RWMol(self._rdkit)
            for match in sorted(
                self._rdkit.GetSubstructMatches(NON_POLAR_H_STRUCT, maxMatches=_MAX_MATCHES),
                reverse=True,
            ):
                editable.RemoveAtom(match[0])
            self._rdkit = editable.GetMol()
        elif hydrogens != "keep":
            raise ValueError(
                f"hydrogens must be 'keep', 'add', 'remove' or 'polar', not {hydrogens!r}."
            )
        self._fix_mol_valence(sanitize=False)  # also the ring information of the new molecule

        # The center atom is chosen from the conformer, so a molecule without one (e.g. from a
        # SMILES string) needs it embedded first.
        if self._rdkit.GetNumConformers() == 0:
            self.__embed()

        # TODO: this doesnt make sense for proteins. In a protein, all torsions should be oriented wrt the backbone, not the center atom.
        rotatable_bonds = self.__find_rotatable_bonds(flex_hydrogens) if flexible else []
        self._center_atom = (
            center_atom if center_atom is not None else self.__get_center_atom(rotatable_bonds)
        )
        # The geometry (bond lengths and angles) that poses are applied to, see `pose_to_positions`.
        # Torsions are absolute and the rotation is relative to this orientation, so the global
        # conformer can be moved afterwards without changing what a pose means.
        self._reference_positions = np.array(self._rdkit.GetConformer().GetPositions(), dtype=float)

        # TODO: make property. Setting it will then compute the torsions if needed. Perhaps bool | list ?
        self.is_flexible = flexible
        self._flex_hydrogens = flex_hydrogens
        if self.is_flexible:
            self.__compute_rotatable_torsions(flex_hydrogens)

        self.__assign_atom_types()

        # Which atoms scoring functions should consider — computed once, here, so every
        # scoring function built on this Mol agrees, instead of each one recomputing (and
        # potentially disagreeing about) the same mask from the same atom_types.
        self.scoring_mask = ~(
            ignore_hydrogens
            & (
                (self._atom_types == AtomType.Hydrogen)
                | (self._atom_types == AtomType.PolarHydrogen)
            )
        )

        Chem.rdPartialCharges.ComputeGasteigerCharges(self._rdkit)

        # Set the layout
        self.layout: PoseLayout = PoseLayout(rotation_type, len(self.__rotatable_torsions))

        self.__cur_rotation = np.zeros(self.layout.rot_dim)

        # If more than 1000 heavy atoms: assume protein for viewing
        if self._rdkit.GetNumHeavyAtoms() >= VIEWER_PROTEINS_HEAVY_ATOMS_CUTOFF:
            self.draw_options = self.__default_protein_draw_options.copy()
        else:
            self.draw_options = self.__default_ligand_draw_options.copy()

    @classmethod
    def from_smiles(
        cls,
        smiles: str,
        hydrogens: Literal["keep", "add", "remove", "polar"] = "polar",
        **kwargs,
    ):
        """Constructs an instance of :class:`Mol` from a SMILES string representation of a molecule.

        Parameters
        ----------
        smiles : str
            The SMILES string representation of the molecule.
        hydrogens : {'keep', 'add', 'remove', 'polar'}, default 'polar'
            Whether to keep hydrogens as is, add additional hydrogens, remove all hydrogens, or
            keep only the polar ones.

        Returns
        -------
        Mol
        """
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        return cls(mol, hydrogens=hydrogens, **kwargs)

    @classmethod
    def from_rdkit(
        cls,
        mol: Chem.Mol,
        hydrogens: Literal["keep", "add", "remove", "polar"] = "polar",
        **kwargs,
    ):
        """Create an instance of :class:`Mol` from an RDKit :class:`~rdkit.Chem.rdchem.Mol` object.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            The input molecule as an RDKit :class:`~rdkit.Chem.rdchem.Mol` object.
        hydrogens : {'keep', 'add', 'remove', 'polar'}, default 'polar'
            Whether to keep hydrogens as is, add additional hydrogens, remove all hydrogens, or
            keep only the polar ones.

        Returns
        -------
        Mol
        """
        return cls(mol, hydrogens=hydrogens, **kwargs)

    @classmethod
    def from_pdb(
        cls,
        pdb_file: str,
        hydrogens: Literal["keep", "add", "remove", "polar"] = "polar",
        template_smiles: str = None,
        template_sdf: str = None,
        **kwargs,
    ):
        r"""Creates an instance of :class:`Mol` from a PDB file.

        Always sanitizes if template included.

        .. warning::
            When loading small molecules from PDB, always include a template.
            Molecules without templates will not be sanitized, and can thus not be used
            in certain scoring functions.


        Parameters
        ----------
        pdb_file : str
            Path to the PDB file containing the molecule.
        hydrogens : {'keep', 'add', 'remove', 'polar'}, default 'polar'
            Whether to keep hydrogens as is, add additional hydrogens, remove all hydrogens, or
            keep only the polar ones.
        template_smiles : str, optional
            SMILES string representing a reference molecule.
            To sanitize the molecule, either `template_smiles` or
            `template_sdf` must be provided.
        template_sdf : str, optional
            Path to an SDF file containing a reference molecule.
            To sanitize the molecule, either `template_smiles` or
            `template_sdf` must be provided.

        Returns
        -------
        Mol
        """
        mol = Chem.MolFromPDBFile(pdb_file, sanitize=False, removeHs=(hydrogens == "remove"))

        if mol is None:
            raise ValueError(f"RDKit could not parse PDB file: {pdb_file}")

        # Strip residues whose atoms have impossible valence (e.g. metal-coordinated O,3
        # or modified residues that survive non-standard replacement).
        mol.UpdatePropertyCache(strict=False)
        problems = Chem.DetectChemistryProblems(mol)
        bad_residues = {
            (info.GetChainId(), info.GetResidueNumber(), info.GetInsertionCode())
            for p in problems
            if hasattr(p, "GetAtomIdx")
            for info in [mol.GetAtomWithIdx(p.GetAtomIdx()).GetPDBResidueInfo()]
            if info is not None
        }
        if bad_residues:
            labels = ", ".join(f"chain {c} res {r}{i.strip()}" for c, r, i in sorted(bad_residues))
            warnings.warn(
                f"from_pdb: removed {len(bad_residues)} residue(s) with impossible valence "
                f"({labels}). Check the source PDB for non-standard or metal-coordinated residues.",
                UserWarning,
                stacklevel=2,
            )
            edit = Chem.RWMol(mol)
            for idx in sorted(
                [
                    a.GetIdx()
                    for a in mol.GetAtoms()
                    if a.GetPDBResidueInfo()
                    and (
                        a.GetPDBResidueInfo().GetChainId(),
                        a.GetPDBResidueInfo().GetResidueNumber(),
                        a.GetPDBResidueInfo().GetInsertionCode(),
                    )
                    in bad_residues
                ],
                reverse=True,
            ):
                edit.RemoveAtom(idx)
            mol = edit.GetMol()

        if template_smiles or template_sdf:
            if template_smiles:
                template_mol = Chem.MolFromSmiles(template_smiles)
            else:
                template_mol = Chem.MolFromMolFile(template_sdf)

            mol = Chem.AllChem.AssignBondOrdersFromTemplate(template_mol, mol)

            Chem.AssignStereochemistryFrom3D(mol)
            Chem.SanitizeMol(mol)

        return cls(mol, hydrogens=hydrogens, **kwargs)

    @classmethod
    def from_sdf(
        cls,
        mol_file: str,
        hydrogens: Literal["keep", "add", "remove", "polar"] = "polar",
        **kwargs,
    ):
        """Creates an instance of :class:`Mol` from an SDF file.

        Parameters
        ----------
        mol_file : str
            Path to the SDF file containing the molecule.
        hydrogens : {'keep', 'add', 'remove', 'polar'}, default 'polar'
            Whether to keep hydrogens as is, add additional hydrogens, remove all hydrogens, or
            keep only the polar ones.

        Returns
        -------
        Mol
        """
        RDLogger.DisableLog("rdApp.*")

        mol = Chem.MolFromMolFile(
            mol_file,
            removeHs=(hydrogens == "remove"),
            strictParsing=False,
            sanitize=False,
        )
        RDLogger.EnableLog("rdApp.*")
        return cls(mol, hydrogens=hydrogens, **kwargs)

    def __embed(self) -> None:
        """Give the molecule a 3D conformer.

        RDKit embeds a molecule far better with its hydrogens, so a molecule without explicit
        hydrogens (``hydrogens='keep'`` or ``'remove'`` of a SMILES string) is embedded with
        temporary ones, and only the coordinates of its own atoms are kept.
        """
        Chem.SanitizeMol(self._rdkit)
        n_atoms = self._rdkit.GetNumAtoms()
        embedded = Chem.AddHs(self._rdkit)  # the added hydrogens come after the atoms already there
        params = Chem.AllChem.ETKDGv3()
        params.randomSeed = 0xC0FFEE
        if Chem.AllChem.EmbedMolecule(embedded, params) == -1:
            raise ValueError("RDKit could not embed a 3D conformer of the molecule.")

        conformer = Chem.Conformer(n_atoms)
        conformer.SetPositions(embedded.GetConformer().GetPositions()[:n_atoms])
        conformer.Set3D(True)
        self._rdkit.AddConformer(conformer, assignId=True)

    def _fix_mol_valence(self, sanitize=True):
        Chem.SanitizeMol(
            self._rdkit,
            sanitizeOps=(Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES),
        )

        for atom in self._rdkit.GetAtoms():
            # print(
            #     atom.GetSymbol(),
            #     atom.GetValence(Chem.ValenceType.EXPLICIT),
            #     atom.GetFormalCharge(),
            # )

            if atom.GetSymbol() == "N" and atom.GetValence(Chem.ValenceType.EXPLICIT) == 4:
                atom.SetFormalCharge(+1)
            if atom.GetSymbol() == "O" and atom.GetValence(Chem.ValenceType.EXPLICIT) == 1:
                atom.SetFormalCharge(-1)
            # incorrect epoxide fix TODO: check with David
            if atom.GetSymbol() == "O" and atom.GetValence(Chem.ValenceType.EXPLICIT) == 2:
                atom.SetFormalCharge(0)

        self._rdkit.UpdatePropertyCache(strict=sanitize)

        if sanitize:
            Chem.SanitizeMol(self._rdkit)
        return self

    # endregion

    # region Chemistry / topology analysis

    def __assign_atom_types(self):
        self._atom_types = np.array([AtomType.Unknown] * len(self._rdkit.GetAtoms()))

        hba = [m[0] for m in self._rdkit.GetSubstructMatches(HBA_STRUCT, maxMatches=_MAX_MATCHES)]

        non_polar_h = [
            m[0]
            for m in self._rdkit.GetSubstructMatches(NON_POLAR_H_STRUCT, maxMatches=_MAX_MATCHES)
        ]

        for i, atom in enumerate(self._rdkit.GetAtoms()):
            atomic_number = atom.GetAtomicNum()

            a_str = atom.GetSymbol()
            if atomic_number == 1 and atom.GetIdx() not in non_polar_h:
                a_str = "HD"
            elif atomic_number == 6 and atom.GetIsAromatic():
                a_str = "A"
            elif atomic_number == 8:
                a_str = "OA"
            elif atomic_number == 7 and atom.GetIdx() in hba:
                a_str = "NA"
            elif atomic_number == 16 and atom.GetIdx() in hba:
                a_str = "SA"

            a_type = AtomType.GenericMetal
            # Assign atom type
            for _, t in vina_atom_consts.items():
                if a_str == t.ad_name:
                    a_type = t.type
                    break

            hbonded = False
            heterobonded = False

            # Get hbonded and heterobonded
            for neigh in atom.GetNeighbors():
                if neigh.GetSymbol() == "H":
                    hbonded = True
                elif neigh.GetSymbol() != "C":
                    heterobonded = True

            a_type = a_type.adjust(hbonded, heterobonded)

            self._atom_types[i] = a_type

    def __get_center_atom(self, rotatable_bonds: list) -> int:
        """Choose the center atom: the pivot of the rotation, and the fixed side of every torsion.

        The side of a rotatable bond that does not contain the center atom is the one that moves,
        so with torsions the center atom is chosen to keep the largest moved fragment small: a
        torsion then never drags most of the molecule along, and the angles are comparably
        sized steps. The choice comes from the molecular graph, so it does not depend on which
        conformer was loaded. Ties are broken by (in order) the fewest atoms moved in total,
        the smallest summed topological distance to the other heavy atoms (the most central), the
        smallest distance to the centroid of the loaded conformer, and the lowest index.

        Without torsions (rigid molecules, and everything that is not ``flexible``, such as a
        receptor) it is the heavy atom closest to the centroid.
        """
        rd = self._rdkit
        heavy = np.array([atom.GetAtomicNum() > 1 for atom in rd.GetAtoms()])
        positions = rd.GetConformer().GetPositions()
        distance = np.linalg.norm(positions - positions[heavy].mean(axis=0), axis=1)
        candidates = np.flatnonzero(heavy)

        if not rotatable_bonds:
            return int(candidates[np.lexsort((candidates, distance[candidates]))[0]])

        adjacency = [[n.GetIdx() for n in atom.GetNeighbors()] for atom in rd.GetAtoms()]
        n_atoms, n_bonds = rd.GetNumAtoms(), len(rotatable_bonds)
        side_a = np.zeros((n_bonds, n_atoms), dtype=bool)  # the atoms on the first atom's side
        side_b = np.zeros((n_bonds, n_atoms), dtype=bool)
        for k, (a, b) in enumerate(rotatable_bonds):
            side_a[k, _atoms_beyond(adjacency, a, b)] = True
            side_b[k, _atoms_beyond(adjacency, b, a)] = True
        # heavy atoms that every bond moves, for every candidate center atom
        moved = np.where(
            side_a, (side_b & heavy).sum(axis=1)[:, None], (side_a & heavy).sum(axis=1)[:, None]
        )[:, candidates]
        central = np.asarray(Chem.GetDistanceMatrix(rd))[:, heavy].sum(axis=1)[candidates]

        order = np.lexsort(
            (candidates, distance[candidates], central, moved.sum(axis=0), moved.max(axis=0))
        )
        return int(candidates[order[0]])

    def __find_rotatable_bonds(self, flex_hydrogens: bool = False) -> list[tuple[int, int]]:
        """The rotatable bonds, as pairs of atom indices.

        Which bonds rotate does not depend on the center atom. Unless `flex_hydrogens`, bonds to
        a terminal heavy atom (a methyl group, a hydroxyl hydrogen, ...) are left out.
        """
        bonds = []
        for match in self._rdkit.GetSubstructMatches(
            ROTATABLE_BOND_STRUCT, maxMatches=_MAX_MATCHES
        ):
            atom_1, atom_2 = match[0], match[1]
            if not flex_hydrogens:
                heavy_degrees = [
                    sum(
                        nbr.GetAtomicNum() > 1
                        for nbr in self._rdkit.GetAtomWithIdx(i).GetNeighbors()
                    )
                    for i in (atom_1, atom_2)
                ]
                if 1 in heavy_degrees:
                    continue
            bonds.append((atom_1, atom_2))
        return bonds

    def __compute_rotatable_torsions(self, flex_hydrogens: bool = False):
        """
        Calculates the rotatable torsion angles and stores them along with their
        indices defining the torsion in the molecule. This includes identifying
        rotatable bonds, constructing torsion definitions, and determining torsion
        angles for each rotatable bond in the molecule. The results are stored as
        attributes for later use.

        """
        rotatable_torsions = []
        rotatable_bonds = self.__find_rotatable_bonds(flex_hydrogens)

        distance_matrix = np.array(Chem.GetDistanceMatrix(self._rdkit))[self._center_atom, :]

        for i_atom_1, i_atom_2 in rotatable_bonds:
            atom_1_neighbors = self._rdkit.GetAtomWithIdx(i_atom_1).GetNeighbors()
            atom_2_neighbors = self._rdkit.GetAtomWithIdx(i_atom_2).GetNeighbors()

            ix_atom_1_neighbors = [a.GetIdx() for a in atom_1_neighbors if a.GetIdx() != i_atom_2]
            ix_atom_2_neighbors = [a.GetIdx() for a in atom_2_neighbors if a.GetIdx() != i_atom_1]

            torsion = (
                min(ix_atom_1_neighbors),
                i_atom_1,
                i_atom_2,
                min(ix_atom_2_neighbors),
            )

            # (a, b, c, d)  |    o (center atom)
            # Als center_atom dichter bij b -> draait niet.
            # Als center_atom dichter bij c -> draait wel -> invert torsion.
            # print(distance_matrix[torsion[1]], distance_matrix[torsion[2]])
            if distance_matrix[torsion[2]] < distance_matrix[torsion[1]]:
                torsion = torsion[::-1]

            rotatable_torsions.append(torsion)

        self.__rotatable_torsions = rotatable_torsions

        adjacency = [[n.GetIdx() for n in atom.GetNeighbors()] for atom in self._rdkit.GetAtoms()]
        self.__torsion_moving = [
            _atoms_beyond(adjacency, c, b) for (_, b, c, _) in rotatable_torsions
        ]
        self.__packed_torsions = _pack_torsions(rotatable_torsions, self.__torsion_moving)

    @property
    def rdkit(self) -> Chem.Mol:
        """The underlying :class:`~rdkit.Chem.rdchem.Mol`, for any RDKit functionality.

        This is the molecule Pyrite works on, not a copy: treat it as read-only (descriptors,
        substructure matches, force fields, ...). To change the molecule, edit a copy
        (:meth:`to_rdkit`) and build a new :class:`Mol` from it.
        """
        return self._rdkit

    def to_rdkit(self, pose: Pose | NDArray | None = None) -> Chem.Mol:
        """A copy of the underlying :class:`~rdkit.Chem.rdchem.Mol`, free to use and to edit.

        Without a `pose` the copy has all the conformers of this molecule. With one, it has a
        single conformer (the default one, ``confId=-1``) with the atoms where the pose puts
        them, so any RDKit function can be applied to a pose without touching this molecule::

            posed = mol.to_rdkit(pose)
            Chem.rdMolTransforms.ComputeCentroid(posed.GetConformer())

        To use an edited copy with Pyrite, build a new :class:`Mol` from it: ``Mol(edited)``.

        Parameters
        ----------
        pose : Pose or array_like, optional
            One pose, or its raw values. A batch of poses is not accepted: call this for every
            pose.

        Returns
        -------
        rdkit.Chem.rdchem.Mol

        Raises
        ------
        ValueError
            When `pose` is a batch.
        """
        if pose is None:
            return Chem.Mol(self._rdkit)
        positions = self.pose_to_positions(pose)
        if positions.ndim != 2:
            raise ValueError("to_rdkit takes one pose, not a batch: call it for every pose.")
        return self._rdkit_with_positions(positions)

    def _rdkit_with_positions(self, positions: NDArray) -> Chem.Mol:
        """A private copy with only the default conformer, set to `positions` ``(n_atoms, 3)``."""
        copy = Chem.Mol(self._rdkit, False, self._rdkit.GetConformer().GetId())
        copy.GetConformer().SetPositions(positions)
        return copy

    @property
    def n_atoms(self) -> int:
        """The number of atoms in the molecule."""
        return self._rdkit.GetNumAtoms()

    @property
    def atoms(self):
        """The :class:`~rdkit.Chem.rdchem.Atom` objects of the molecule, in index order."""
        return self._rdkit.GetAtoms()

    @property
    def n_conformers(self) -> int:
        """The number of conformers: the global one, and any made by :meth:`pose_to_conformer`."""
        return self._rdkit.GetNumConformers()

    def remove_conformer(self, conf_id: int) -> None:
        """Remove a conformer, e.g. one made by :meth:`pose_to_conformer` with ``new_conf=True``."""
        self._rdkit.RemoveConformer(conf_id)

    @property
    def atom_types(self):
        """The atom types of all atoms in the ligand.

        Returns
        -------
        numpy.ndarray
        """
        return self._atom_types

    # endregion

    # region Pose & geometry

    @property
    def positions(self):
        """The positions of all atoms in the global conformer.

        Returns
        -------
        list
        """
        return self._rdkit.GetConformer().GetPositions()

    def get_positions(self, conf_id: int = -1, poses=None) -> NDArray[np.float32]:
        """Returns the positions of all atoms in a specific conformer.

        Parameters
        ----------
        conf_id : int, default -1
             The conformer id to retrieve positions from. By default selects the global conformer.

        Returns
        -------
        list
        """
        if poses is not None:
            return self.pose_to_positions(poses)
        return self._rdkit.GetConformer(conf_id).GetPositions()

    def pose_to_positions(self, poses) -> NDArray:
        """Compute the atom positions of poses, in numpy, without any RDKit conformer.

        This is exactly what :meth:`pose_to_conformer` does to a conformer, but it returns the positions
        directly, for one pose or a whole batch at once. The rotation is applied about the center
        atom, the center atom is placed at the translation, and then the torsions are set (as
        ``pose_to_conformer`` does, in that order).

        Parameters
        ----------
        poses : Pose or Poses or array_like
            The poses, in the layout of this molecule. A raw array is a single pose of shape
            ``(n_dims,)`` or a batch of shape ``(n, n_dims)``.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_atoms, 3)`` for a single pose, or ``(n, n_atoms, 3)`` for a batch.
        """
        if isinstance(poses, Pose | Poses):
            assert poses.layout == self.layout, "Pose and molecule layout do not match."
            values = np.asarray(poses)
        else:
            values = np.asarray(poses, dtype=float)
        single = values.ndim == 1
        values = np.atleast_2d(values)
        if values.shape[-1] != self.layout.n_dims:
            raise ValueError(
                f"Poses for this molecule have {self.layout.n_dims} variables, got {values.shape[-1]}."
            )
        layout = self.layout
        if len(values) == 0:  # SciPy cannot build an empty set of rotations
            return np.empty((0, *self._reference_positions.shape))

        reference = self._reference_positions
        centered = reference - reference[self._center_atom]
        positions = np.einsum(
            "nij,aj->nai", layout.rotation_matrix(values[:, layout.rot_slice]), centered
        )
        positions += values[:, None, layout.trans_slice]
        if self.n_tors:
            positions = _apply_torsions(
                positions, self.__packed_torsions, values[:, layout.tors_slice]
            )
        return positions[0] if single else positions

    @property
    def rotatable_torsions(self):
        """The rotatable torsions of the molecule.

        Returns
        -------
        list
            A list containing all rotatable torsions in the molecule,
            indicated by four atom indices. An entry looks like
            ``[i, j, k, l]``, where the rotated bond is between
            ``j`` and ``k``, and all atoms attached to ``k`` are moved.

        """
        return self.__rotatable_torsions

    @property
    def torsions(self) -> NDArray[np.float64]:
        """The current torsion angles of the global conformer, one per rotatable torsion.

        A new array on every call, in the order of :attr:`rotatable_torsions`, and empty for a
        molecule that is not ``flexible``.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_tors,)``, the angles in radians.
        """
        conformer = self._rdkit.GetConformer()
        return np.array(
            [
                Chem.rdMolTransforms.GetDihedralRad(conformer, *torsion)
                for torsion in self.__rotatable_torsions
            ],
            dtype=float,
        )

    @property
    def n_tors(self):
        """The number of torsions in the molecule."""
        return self.layout.n_tors

    def pose_to_conformer(self, pose: Pose | NDArray, new_conf: bool = False) -> int:
        """Put a pose on a conformer of the molecule.

        The positions are those of :meth:`pose_to_positions`, set on either the global
        conformer, or on a new conformer.

        .. note::
            Scoring does not use this method: scoring functions get positions from
            :meth:`pose_to_positions`, or a private RDKit copy, and never modify the molecule.
            This changes the molecule (a conformer of it), so it is for export and display.

        Parameters
        ----------
        pose : Pose or array_like
            The pose, or its values ``(roll, pitch, yaw, x, y, z, *torsions)`` (the rotation
            has four values for a quaternion layout).
        new_conf : bool, default False
            Whether to put the pose on a new conformer, instead of on the global conformer.

        Returns
        -------
        int
            Conformer id of the new conformer. If no new conformer is created, returns -1,
            which is the id of the global conformer.

        """
        if isinstance(pose, Pose):
            assert pose.layout == self.layout, "Pose and molecule layout do not match."
        else:
            assert len(pose) == self.layout.rot_dim + 3 + self.layout.n_tors, (
                "Pose parameterization does not match molecule layout."
            )
            pose = Pose(pose, self.layout)

        positions = self.pose_to_positions(pose)
        if new_conf:
            conformer = Chem.Conformer(self._rdkit.GetConformer())  # a copy, with its flags
            conformer.SetPositions(positions)
            return self._rdkit.AddConformer(conformer, assignId=True)

        self._rdkit.GetConformer().SetPositions(positions)
        self.__cur_rotation = np.array(pose.rotation, dtype=float)
        return -1

    @property
    def center_atom(self):
        """The index of the atom used as center of the molecule.

        Returns
        -------
        int
        """
        return self._center_atom

    @center_atom.setter
    def center_atom(self, value):
        # Rotations are about the center atom, so the global conformer is put back in the reference
        # orientation (keeping its torsions), with the new center atom at the origin.
        self._center_atom = value
        if self.is_flexible:
            # Which side of every torsion moves depends on the center atom: it must stay fixed,
            # or a pose's translation is no longer where the center atom ends up. The torsion
            # values and their order do not change, so existing poses stay valid.
            self.__compute_rotatable_torsions(self._flex_hydrogens)
        pose = np.concatenate([self.layout.identity_rotation, np.zeros(3), self.torsions])
        positions = self.pose_to_positions(pose)
        positions -= positions[value]  # the torsions can move the new center atom
        self._rdkit.GetConformer().SetPositions(positions)
        self.__cur_rotation = np.array(self.layout.identity_rotation, dtype=float)

    @property
    def position(self):
        """The position of the molecule.

        This is equal to the position of the center atom.

        Returns
        -------
        list
            A list of shape (3,) containing the x, y, and z coordinates of the center atom.
        """
        center_atom_coords = self._rdkit.GetConformer().GetAtomPosition(self._center_atom)
        return [center_atom_coords.x, center_atom_coords.y, center_atom_coords.z]

    @property
    def rotation(self):
        """The current rotation of the molecule.

        This is the roll, pitch, and yaw angles of global conformer of the molecule.

        Returns
        -------
        list
            A list of shape (3,) containing the roll, pitch, and yaw angles of the molecule.
        """
        return self.__cur_rotation

    # endregion

    # region Conformers

    def get_n_conformer_torsion_configurations(self, n: int, seed: int = 0xC0FFEE) -> NDArray:
        """Retrieve the torsions of `n` conformers of the molecule.

        The conformers are generated with RDKit's ETKDG, so they are physically realistic. For
        independent random torsions, which can be physically impossible, see
        :meth:`PoseLayout.sample_random_torsions`.

        .. note::
            This method does not necessarily result in unique configurations, and RDKit can
            return fewer than `n` conformers.

        Parameters
        ----------
        n : int
            The number of conformers to generate.
        seed : int, default 0xC0FFEE
            The random seed of the conformer generation.

        Returns
        -------
        numpy.ndarray
            An array of shape ``(n_conformers, n_tors)`` with the torsion angles of every
            conformer, in radians.
        """
        params = Chem.AllChem.ETKDGv3()
        params.randomSeed = seed

        # Embedded with temporary hydrogens, like the molecule itself (see `__embed`): a molecule
        # without them embeds worse, and RDKit can refuse to embed one with stereocentres. The
        # atoms of this molecule keep their indices, so the torsions are read off as they are.
        new_mol = Chem.AddHs(self._rdkit)

        cids = Chem.AllChem.EmbedMultipleConfs(new_mol, n, params)

        if len(cids) == 0:
            raise ValueError("RDKit could not embed any conformers of the molecule.")

        # EmbedMultipleConfs can return fewer than `n` conformers.
        configurations = np.empty((len(cids), len(self.__rotatable_torsions)))
        for i, cid in enumerate(cids):
            torsion_angles = np.zeros(len(self.__rotatable_torsions))
            for j, torsion in enumerate(self.__rotatable_torsions):
                torsion_angles[j] = Chem.rdMolTransforms.GetDihedralRad(
                    new_mol.GetConformer(cid), *torsion
                )
            configurations[i] = torsion_angles

        return configurations

    # endregion

    # region Export

    def to_sdf(
        self,
        file: str | SDWriter,
        poses: Pose | Poses | None = None,
        conf_id: int = -1,
    ) -> None:
        """Write the molecule to an SDF file: one record for a pose, one per pose of ``Poses``.

        Writing poses does not touch the molecule.

        Parameters
        ----------
        file : str, ~rdkit.Chem.rdmolfiles.SDWriter
            Either a filename of a file to create, or an open
            :class:`~rdkit.Chem.rdmolfiles.SDWriter` object to write to (which is left open).
        poses : Pose or Poses, optional
            The pose(s) to write. By default the conformer `conf_id` of the molecule is written.
        conf_id : int, default -1
            The conformer id to write when no `poses` are given. By default the global conformer.

        Raises
        ------
        ValueError
            When both `poses` and a `conf_id` are given.
        """
        if poses is not None and conf_id != -1:
            raise ValueError("Give either `poses` or a `conf_id`, not both.")

        close_writer = False
        if isinstance(file, str):
            writer = SDWriter(file)
            close_writer = True
        elif isinstance(file, SDWriter):
            writer = file
        else:
            raise ValueError("file must be either filename str or SDWriter")

        if poses is None:
            writer.write(self._rdkit, confId=conf_id)
        else:
            positions = self.pose_to_positions(poses)
            if positions.ndim == 2:  # a single pose
                positions = positions[None]
            # a private copy with only the global conformer, so the molecule is left as it is
            work = Chem.Mol(self._rdkit, False, self._rdkit.GetConformer().GetId())
            for pose_positions in positions:
                work.GetConformer().SetPositions(pose_positions)
                writer.write(work)

        if close_writer:
            writer.close()

    def poses_from_sdf(self, file: str, rmsd_delta: float = 0.5) -> Poses:
        """Read the conformers in an SDF file as poses of this molecule.

        The torsions of every conformer are measured, and the rotation and translation are the
        rigid fit of this molecule's reference geometry (with those torsions) onto it. The
        conformers must be of this molecule, with the same atoms in the same order.

        Parameters
        ----------
        file : str
            Path to an SDF file with one or more conformations of this molecule, for example
            written by :meth:`to_sdf`, or by a docking program.
        rmsd_delta : float, default 0.5
            The largest RMSD (in Angstrom) allowed between a conformer and the pose that is made
            of it. Bond lengths and angles are not part of a pose, so a program that changes them
            gives conformers that a pose cannot reproduce exactly; this is how much is accepted.

        Returns
        -------
        Poses
            In the layout of this molecule, one pose per record, in file order.

        Raises
        ------
        ValueError
            When a record is not this molecule (other atoms, bonds or order), or cannot be
            reproduced within `rmsd_delta`.
        OSError
            When the file is missing or empty (RDKit's own error).
        """
        layout = self.layout
        template = self._rdkit
        bonds = {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in template.GetBonds()}
        elements = [a.GetAtomicNum() for a in template.GetAtoms()]

        values = []
        work = Chem.Mol(template, False, template.GetConformer().GetId())
        for record in self.__read_sdf_records(file):
            same_molecule = [a.GetAtomicNum() for a in record.GetAtoms()] == elements and {
                frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in record.GetBonds()
            } == bonds
            if not same_molecule:
                raise ValueError(
                    f"Record {len(values)} of {file} is not this molecule (or its atoms are in a "
                    "different order)."
                )
            observed = record.GetConformer().GetPositions()

            work.GetConformer().SetPositions(observed)
            torsions = np.array(
                [
                    Chem.rdMolTransforms.GetDihedralRad(work.GetConformer(), *torsion)
                    for torsion in self.__rotatable_torsions
                ]
            )
            # the reference geometry with these torsions, the center atom at the origin
            identity = np.concatenate([layout.identity_rotation, np.zeros(3), torsions])
            shape = self.pose_to_positions(identity)

            # the rigid motion that takes it onto the conformer (Kabsch)
            shape_mean, observed_mean = shape.mean(axis=0), observed.mean(axis=0)
            u, _, vt = np.linalg.svd((shape - shape_mean).T @ (observed - observed_mean))
            reflection = np.sign(np.linalg.det(vt.T @ u.T))
            rotation = vt.T @ np.diag([1.0, 1.0, reflection]) @ u.T
            translation = observed_mean - rotation @ shape_mean

            rmsd = np.sqrt(np.mean(np.sum((shape @ rotation.T + translation - observed) ** 2, 1)))
            if rmsd > rmsd_delta:
                raise ValueError(
                    f"Record {len(values)} of {file} cannot be reproduced as a pose: RMSD "
                    f"{rmsd:.2f} > rmsd_delta {rmsd_delta}."
                )
            values.append(
                np.concatenate([layout.rotation_from_matrix(rotation), translation, torsions])
            )

        if not values:
            raise ValueError(f"No molecules found in {file}.")
        return Poses(np.array(values), layout)

    def __read_sdf_records(self, file: str) -> list[Chem.Mol]:
        """The records of an SDF file, with or without hydrogens, as many as this molecule has."""
        for remove_hs in (False, True):
            records = [
                m
                for m in Chem.SDMolSupplier(file, removeHs=remove_hs, sanitize=False)
                if m is not None
            ]
            if records and records[0].GetNumAtoms() == self.n_atoms:
                return records
        raise ValueError(f"The molecules in {file} do not have {self.n_atoms} atoms.")

    # endregion

    # region Visualization / display

    __default_ligand_draw_options = {
        "protein": False,
        "size": (400, 300),
        "colorPalette": "default",
        "note": "",
        "highlight": "",
        "colorscheme": "default",
    }

    __default_protein_draw_options = {
        "protein": True,
        "size": (400, 300),
        "color": "blue",
        "style": "rectangle",
        "surfacetype": "MS",
        "surfacecolor": "white",
        "surfaceopacity": 0.75,
        "stickresidues": [],
        "hideprotein": False,
        "note": "",
    }

    def set_draw_options(self, options):
        """Set draw options.

        Parameters
        ----------
        options : dictionary
            The options to apply.

        """
        # unknown = set(options) - set(self.draw_options)
        # if unknown:
        #     raise ValueError(f"Unknown options: {unknown}")

        self.draw_options.update(options)

    def _viewer_add_(self, viewer, c_m_id, options: dict = None):
        if options is None:
            options = {}
        l_options = self.draw_options.copy()
        l_options.update(options)

        is_protein = (
            l_options["protein"]
            if l_options["protein"] != "auto"
            else self._rdkit.GetNumHeavyAtoms() >= VIEWER_PROTEINS_HEAVY_ATOMS_CUTOFF
        )

        if is_protein:
            m_id = self._viewer_add_prot(viewer, c_m_id, l_options)
        else:
            mblock = Chem.MolToMolBlock(self._rdkit)
            viewer.view.addModel(mblock, "mol")
            m_id = c_m_id + 1
            viewer.view.setStyle(
                {"model": m_id}, {"stick": {"colorscheme": l_options["colorscheme"]}}
            )

            viewer._set_hover(self, m_id, self.draw_options)  # noqa
        return m_id

    def _viewer_add_prot(self, viewer, c_m_id, options: dict = None):
        pdbblock = Chem.MolToPDBBlock(self._rdkit)

        viewer.view.addModel(pdbblock, "pdb")
        m_id = c_m_id + 1
        viewer.view.setStyle(
            {"model": m_id},
            {
                "cartoon": {
                    "color": options["color"],
                    "style": options["style"],
                    "hidden": options["hideprotein"],
                },
            },
        )

        for res in options["stickresidues"]:
            viewer.view.setStyle(
                {"resn": res, "byres": "true"},
                {"stick": {"colorscheme": "whiteCarbon"}},
            )

        surf = True
        surface_type = py3Dmol.MS
        match options["surfacetype"]:
            case "MS":
                surface_type = py3Dmol.MS
            case "VDW":
                surface_type = py3Dmol.VDW
            case "SAS":
                surface_type = py3Dmol.SAS
            case "SES":
                surface_type = py3Dmol.SES
            case _:
                surf = False

        if surf:
            viewer.view.addSurface(
                surface_type,
                {
                    "opacity": options["surfaceopacity"],
                    "color": options["surfacecolor"],
                },
                {"model": m_id},
            )

        if "note" in options:
            viewer._set_hover(self, m_id, options)

        return m_id

    def _repr_png_(self):
        return self.__repr_picture(Draw.MolDraw2DCairo(*self.draw_options["size"]))

    def _repr_svg_(self):
        return self.__repr_picture(Draw.MolDraw2DSVG(*self.draw_options["size"]))

    def _ipython_display_(self):
        display(
            Viewer(  # noqa
                self,
                width=self.draw_options["size"][0],
                height=self.draw_options["size"][1],
            ).as_widget()
        )

    @property
    def png(self):
        """A png of this ligand for use in jupyter notebooks.

        Returns
        -------
        ~IPython.display.Image
        """
        return Image(self._repr_png_(), embed=True)

    @property
    def svg(self):
        """A svg of this ligand for use in jupyter notebooks.

        Returns
        -------
        ~IPython.display.SVG
        """
        return SVG(self._repr_svg_())

    @property
    def viewer(self):
        """A viewer containing this ligand.

        Returns
        -------
        Viewer
        """
        return Viewer(
            self,
            width=self.draw_options["size"][0],
            height=self.draw_options["size"][1],
        )

    def __repr_picture(self, d2d):
        dopts = d2d.drawOptions()

        if "colorPalette" in self.draw_options:
            match self.draw_options["colorPalette"].lower():
                case "cdk":
                    dopts.useCDKAtomPalette()
                case "bw":
                    dopts.useBWAtomPalette()
                case "avalon":
                    dopts.useAvalonAtomPalette()

        if "note" not in self.draw_options:
            self.draw_options["note"] = ""
        match self.draw_options["note"].lower():
            case "idx":
                for a in self._rdkit.GetAtoms():
                    a.SetProp("atomNote", f"{a.GetIdx()}")
            case "type":
                for a in self._rdkit.GetAtoms():
                    ty = self._atom_types[a.GetIdx()]
                    a.SetProp("atomNote", f"{str(ty)}")
            case _:
                for a in self._rdkit.GetAtoms():
                    a.ClearProp("atomNote")

        highlight = []
        if "highlight" in self.draw_options:
            match self.draw_options["highlight"].lower():
                case "center":
                    highlight = [self._center_atom]
                case list():
                    highlight = self.draw_options["highlight"]

        d2d.DrawMolecule(self._rdkit, highlightAtoms=highlight)
        d2d.FinishDrawing()
        return d2d.GetDrawingText()

    # endregion

    # region Copying

    def copy(self) -> Mol:
        """A full, independent copy: the RDKit molecule with all its conformers, and every
        derived value (reference geometry, torsions, atom types, layout, center atom).

        Nothing is re-derived, so unlike ``Mol(mol.rdkit)`` the copy also keeps the current
        state of the global conformer and the custom settings of this molecule.
        """
        return copy.deepcopy(self)

    # Pickling: RDKit leaves out private properties (like the Gasteiger charges that scoring
    # functions read) and stores coordinates in single precision, unless asked otherwise.
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        default = Chem.GetDefaultPickleProperties()
        Chem.SetDefaultPickleProperties(
            Chem.PropertyPickleOptions.AllProps | Chem.PropertyPickleOptions.CoordsAsDouble
        )
        try:
            state["_rdkit"] = self._rdkit.ToBinary()
        finally:
            Chem.SetDefaultPickleProperties(default)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._rdkit = Chem.Mol(state["_rdkit"])

    def __copy__(self) -> Mol:
        return self.copy()  # a shallow copy would share the RDKit molecule

    def __deepcopy__(self, memo) -> Mol:
        new = type(self).__new__(type(self))
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            new.__dict__[key] = Chem.Mol(value) if key == "_rdkit" else copy.deepcopy(value, memo)
        return new

    # endregion


def _as_array(values: NDArray, dtype, copy) -> NDArray:
    """The ``__array__`` protocol of NumPy 2: the owned array itself unless a copy is needed."""
    if dtype is not None and np.dtype(dtype) != values.dtype:
        if copy is False:
            raise ValueError("A copy is required to convert the dtype, but copy=False was given.")
        return values.astype(dtype)
    return values.copy() if copy else values


@dataclass(frozen=True)
class PoseLayout:
    rot_type: Literal["euler", "quat"]  # 'euler' | 'quat'
    n_tors: int

    # precomputed properties
    rot_dim: int = field(init=False, repr=False, compare=False)
    rot_slice: slice = field(init=False, repr=False, compare=False)
    trans_slice: slice = field(init=False, repr=False, compare=False)
    tors_slice: slice = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        d = 3 if self.rot_type == "euler" else 4 if self.rot_type == "quat" else None
        if d is None:
            raise ValueError(f"Unknown rot_type {self.rot_type!r}. Expected 'euler' or 'quat'.")

        object.__setattr__(self, "rot_dim", d)
        object.__setattr__(self, "rot_slice", slice(0, d))
        object.__setattr__(self, "trans_slice", slice(d, d + 3))
        object.__setattr__(self, "tors_slice", slice(d + 3, None))

    @property
    def n_dims(self) -> int:
        """The length of a pose vector in this layout."""
        return self.rot_dim + 3 + self.n_tors

    @property
    def identity_rotation(self) -> NDArray[np.float32]:
        if self.rot_type == "euler":
            return np.zeros(3)
        return np.array([1, 0, 0, 0], dtype=np.float32)

    def compose_rotation(self, rotation: NDArray, delta_rotvec: NDArray) -> NDArray:
        """Rotate `rotation` by a small world-frame rotation, in this layout's representation.

        Composes on the left (``dR * R``), so `delta_rotvec` is expressed in the fixed frame,
        independent of the current orientation (left-invariant), rather than in Euler-angle
        space where a uniform step is not a uniform rotation.

        Parameters
        ----------
        rotation : ndarray
            Current rotation(s), shape ``(rot_dim,)`` or ``(n, rot_dim)``.
        delta_rotvec : ndarray
            Rotation vector(s) (axis * angle, radians), shape ``(3,)`` or ``(n, 3)``,
            broadcastable against `rotation`.

        Returns
        -------
        ndarray
            The perturbed rotation(s), same shape as `rotation`.
        """
        rotation = np.asarray(rotation)
        d_r = Rotation.from_rotvec(delta_rotvec)
        if self.rot_type == "euler":
            # Layout order is (roll, pitch, yaw) = intrinsic ZYX with angles (yaw, pitch, roll).
            current = Rotation.from_euler("ZYX", rotation[..., ::-1])
            return (d_r * current).as_euler("ZYX")[..., ::-1]
        # Layout order is (w, x, y, z); scipy is (x, y, z, w).
        current = Rotation.from_quat(rotation[..., [1, 2, 3, 0]])
        return (d_r * current).as_quat()[..., [3, 0, 1, 2]]

    def rotation_matrix(self, rotation: NDArray) -> NDArray:
        """Return the rotation matrix of a rotation in this layout's representation.

        Parameters
        ----------
        rotation : ndarray
            Shape ``(rot_dim,)`` or ``(n, rot_dim)``.

        Returns
        -------
        ndarray
            Shape ``(3, 3)`` or ``(n, 3, 3)``. It acts on column vectors: ``x' = R @ x``.
        """
        rotation = np.asarray(rotation)
        if self.rot_type == "euler":
            return Rotation.from_euler("ZYX", rotation[..., ::-1]).as_matrix()
        return Rotation.from_quat(rotation[..., [1, 2, 3, 0]]).as_matrix()

    def rotation_from_matrix(self, matrix: NDArray) -> NDArray:
        """Return the rotation(s) in this layout's representation of rotation matrices.

        The inverse of :meth:`rotation_matrix`. A quaternion is returned with ``w >= 0`` (``q`` and
        ``-q`` are the same rotation).

        Parameters
        ----------
        matrix : ndarray
            Shape ``(3, 3)`` or ``(n, 3, 3)``, acting on column vectors.

        Returns
        -------
        ndarray
            Shape ``(rot_dim,)`` or ``(n, rot_dim)``.
        """
        rotation = Rotation.from_matrix(np.asarray(matrix))
        if self.rot_type == "euler":
            return rotation.as_euler("ZYX")[..., ::-1]
        quaternion = rotation.as_quat()[..., [3, 0, 1, 2]]
        return np.where(quaternion[..., :1] < 0, -quaternion, quaternion)

    def sample_random_rotations(self, n: int, rng: np.random.Generator | None = None) -> NDArray:
        """Sample `n` rotations uniformly at random, in this layout's representation.

        The rotations are uniform on SO(3) (the Haar measure): every orientation is equally
        likely. Note that this is *not* the same as drawing every Euler angle uniformly, which
        over-samples the orientations near the poles (pitch of ±90°).

        Parameters
        ----------
        n : int
            The number of rotations.
        rng : numpy.random.Generator, optional
            The source of randomness. Defaults to a fresh generator.

        Returns
        -------
        numpy.ndarray
            An array of shape ``(n, rot_dim)``.
        """
        rng = np.random.default_rng() if rng is None else rng
        # A unit quaternion uniform on the 3-sphere is a uniform rotation; its normalised
        # Gaussian components are uniform on the sphere. Layout order is (w, x, y, z).
        quat = rng.normal(size=(n, 4))
        quat /= np.linalg.norm(quat, axis=1, keepdims=True)
        if self.rot_type == "quat":
            return quat
        return Rotation.from_quat(quat[:, [1, 2, 3, 0]]).as_euler("ZYX")[:, ::-1]

    def sample_random_torsions(self, n: int, rng: np.random.Generator | None = None) -> NDArray:
        """Sample `n` sets of independent, uniformly random torsion angles.

        .. warning::
            Independent random torsions can be physically impossible (atoms on top of each
            other). For realistic torsions, see :meth:`Mol.get_n_conformer_torsion_configurations`.

        Parameters
        ----------
        n : int
            The number of sets.
        rng : numpy.random.Generator, optional
            The source of randomness. Defaults to a fresh generator.

        Returns
        -------
        numpy.ndarray
            An array of shape ``(n, n_tors)`` with angles in ``[-π, π)``.
        """
        rng = np.random.default_rng() if rng is None else rng
        return rng.uniform(-np.pi, np.pi, size=(n, self.n_tors))


class Pose:
    __slots__ = ("_v", "layout", "rotation", "translation", "torsions")

    def __init__(self, v: NDArray, layout: PoseLayout):
        self._v = np.asarray(v)
        if self._v.shape != (layout.n_dims,):
            raise ValueError(
                f"A pose in {layout} has shape ({layout.n_dims},), got {self._v.shape}."
            )
        self.layout = layout
        self.rotation = self._v[self.layout.rot_slice]
        self.translation = self._v[self.layout.trans_slice]
        self.torsions = self._v[self.layout.tors_slice]

    def __array__(self, dtype=None, copy=None):
        return _as_array(self._v, dtype, copy)

    def __eq__(self, other):
        return bool(np.array_equal(self._v, other._v)) and self.layout == other.layout

    def __getstate__(self):
        # Only _v/layout are real state — rotation/translation/torsions are views
        # derived from them. Pickling them separately (the default for a __slots__
        # class) serializes each view's own buffer as independent data, so they
        # come back as copies, not views, after unpickling. Reconstructing via
        # __init__ in __setstate__ re-derives them as views instead.
        return self._v, self.layout

    def __setstate__(self, state):
        self.__init__(*state)

    @classmethod
    def from_array(cls, v: NDArray, layout: PoseLayout):
        return cls(np.asarray(v).copy(), layout)


class Poses:
    __slots__ = ("_vs", "layout", "rotation", "translation", "torsions")

    def __init__(self, vs: NDArray, layout: PoseLayout):
        self._vs = np.asarray(vs)
        if self._vs.ndim != 2 or self._vs.shape[1] != layout.n_dims:
            raise ValueError(
                f"Poses in {layout} have shape (n, {layout.n_dims}), got {self._vs.shape}."
            )
        self.layout = layout
        self.rotation = self._vs[:, layout.rot_slice]
        self.translation = self._vs[:, layout.trans_slice]
        self.torsions = self._vs[:, layout.tors_slice]

    def __eq__(self, other):
        return bool(np.array_equal(self._vs, other._vs)) and self.layout == other.layout

    def __getstate__(self):
        # See Pose.__getstate__ — same reasoning, same fix.
        return self._vs, self.layout

    def __setstate__(self, state):
        self.__init__(*state)

    def __len__(self):
        return len(self._vs)

    def __array__(self, dtype=None, copy=None):
        return _as_array(self._vs, dtype, copy)

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            return Pose(self._vs[idx], self.layout)
        return Poses(self._vs[idx], self.layout)

    def __iter__(self):
        for row in self._vs:
            yield Pose(row, self.layout)

    @classmethod
    def from_array(cls, vs: NDArray, layout: PoseLayout):
        return cls(np.asarray(vs).copy(), layout)

    @classmethod
    def from_parts(
        cls, layout: PoseLayout, rotation: NDArray, translation: NDArray, torsions: NDArray
    ):
        """Assemble poses from their rotations, translations and torsions.

        Parameters
        ----------
        layout : PoseLayout
            The layout of the poses.
        rotation : array_like
            Shape ``(n, rot_dim)``, in the representation of the layout.
        translation : array_like
            Shape ``(n, 3)``.
        torsions : array_like
            Shape ``(n, n_tors)``.
        """
        vs = np.empty((len(translation), layout.n_dims))
        vs[:, layout.rot_slice] = rotation
        vs[:, layout.trans_slice] = translation
        vs[:, layout.tors_slice] = torsions
        return cls(vs, layout)

    @classmethod
    def from_list(cls, poses: list[Pose], layout: PoseLayout | None = None):
        return cls(np.stack([np.asarray(p) for p in poses]), layout or poses[0].layout)
