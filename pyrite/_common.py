from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import py3Dmol
from IPython.display import SVG, Image
from numpy.typing import NDArray
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, SDWriter, rdDepictor
from scipy.spatial.transform import Rotation

from ._util import (
    _atoms_beyond,
    _pack_torsions,
    _pose_gradient_kernel,
    _pose_positions_kernel,
)
from .atom_consts import AtomType, vina_atom_consts
from .view import Viewer

ROTATABLE_BOND_STRUCT = Chem.MolFromSmarts(
    "[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])"
    "([CH3])[CH3])&!$([CX3](=[N,O,S])-!@[#7,O,S!D1])&!$([#7,O,S!D1]-!@[CX3]"
    "=[N,O,S])&!$([CX3](=[N+])-!@[#7!D1])&!$([#7!D1]-!@[CX3]=[N+])]-,:;!@"
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


def _charge_deprotonated_oxygens(mol: Chem.Mol) -> None:
    """Give a -1 charge to the oxygens of a PDB molecule that have one bond and no hydrogen.

    A PDB file has no charges. When it has hydrogens, an oxygen with a single bond and no
    hydrogen has lost its proton: the carboxylates of Asp, Glu and the C-terminus, a phosphate.
    Without hydrogens in the file, a hydroxyl looks the same, so nothing is changed.
    """
    if not any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
        return
    mol.UpdatePropertyCache(strict=False)  # the valences, after residues may have been removed
    for atom in mol.GetAtoms():
        if (
            atom.GetSymbol() == "O"
            and atom.GetFormalCharge() == 0
            and atom.GetValence(Chem.ValenceType.EXPLICIT) == 1
            and all(n.GetAtomicNum() != 1 for n in atom.GetNeighbors())
        ):
            atom.SetFormalCharge(-1)


class Mol:
    """
    Representation of a molecule.

    The Mol class provides methods for initializing molecules from various sources, such as
    SMILES strings, PDB files, SDF files, or RDKit molecules: the files are read with
    `RDKit <https://www.rdkit.org/docs/>`_. It assigns atom types, rotatable torsions, and the
    center atom of the molecule, and turns poses into atom positions.

    A ``Mol`` holds an RDKit molecule, it is not one: use :attr:`rdkit` for RDKit functionality,
    treating it as read-only, and :meth:`to_rdkit` for a copy to edit.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The RDKit molecule, with a conformer (one is embedded if it has none). It is copied.
    flexible : bool, default False
        Whether the molecule can change its shape in a pose. If ``True``, every rotatable bond
        becomes a torsion: a variable of the poses of this molecule, which a search can turn.
        If ``False``, the molecule is rigid, and a pose only has a rotation and a translation.
    hydrogens : {'keep', 'add', 'remove', 'polar'}, default 'polar'
        Whether to keep hydrogens as is, add the missing hydrogens, remove all hydrogens, or have
        keep only the polar ones (on N, O, S, ...), which decide which atoms are hydrogen bond
        donors. 'polar' adds none: a molecule without hydrogens (a SMILES string, or a file
        without them) has no donors, unless loaded with 'add'.
    ignore_hydrogens : bool, default True
        Whether hydrogen atoms are left out of :attr:`scoring_mask`, the atoms scoring functions
        use. Polar hydrogens still decide which neighbours are donors.
    flex_hydrogens : bool, default False
        Whether bonds to a terminal heavy atom (a methyl or hydroxyl group) are rotatable too.
    center_atom : int, optional
        The index of the atom to use as the center point for rotations. It is also the fixed
        side of every torsion. By default, it is chosen so that no torsion moves a large part of
        the molecule, or, without torsions, the heavy atom closest to the centroid.
    rotation_type : {'euler', 'quat'}, default 'euler'
        How the rotation of a pose is represented: Euler angles, or a quaternion.

    See Also
    --------
    Pose : A pose of a molecule: its rotation, translation and torsions.
    pyrite.scoring.ScoringFunction : Scores poses of a molecule.

    Examples
    --------
    >>> ligand = Mol.from_sdf("ligand.sdf", flexible=True)
    >>> receptor = Mol.from_pdb("receptor.pdb")
    >>> ligand.n_atoms, ligand.n_tors  # the number of atoms and torsions

    A molecule from a SMILES string gets a conformer from RDKit:

    >>> ethanol = Mol.from_smiles("CCO", hydrogens="add")
    >>> ethanol.n_atoms
    9
    """

    # region Construction

    def __init__(
        self,
        mol: Chem.Mol,
        *,
        flexible: bool = False,  # TODO: allow list of resids. No. Read everything rigid, and allow for auto setting of torsions, or manual, or by resid.
        hydrogens: Literal["keep", "add", "remove", "polar"] = "polar",
        ignore_hydrogens: bool = True,
        flex_hydrogens: bool = False,
        center_atom: int = None,
        rotation_type: Literal["euler", "quat"] = "euler",
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
        self.__torsion_offsets = np.empty(0)

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
        # the new molecule needs its ring information; its valences must not be read as charges
        self._fix_mol_valence(sanitize=False, assign_charges=False)

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
        self._scoring_mask = ~(
            ignore_hydrogens
            & (
                (self._atom_types == AtomType.Hydrogen)
                | (self._atom_types == AtomType.PolarHydrogen)
            )
        )

        Chem.rdPartialCharges.ComputeGasteigerCharges(self._rdkit)

        # Set the layout
        self._layout = PoseLayout(rotation_type, len(self.__rotatable_torsions))

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
        *,
        hydrogens: Literal["keep", "add", "remove", "polar"] = "polar",
        **kwargs,
    ):
        """Construct an instance of :class:`Mol` from a SMILES string representation of a molecule.

        A 3D conformer is embedded with RDKit's ETKDG, with temporary hydrogens (an embedding
        without hydrogens is worse). The charges are those in the SMILES string: ``O`` with one bond
        is a hydroxyl.

        Parameters
        ----------
        smiles : str
            The SMILES string representation of the molecule.
        hydrogens : {'keep', 'add', 'remove', 'polar'}, default 'polar'
            Whether to keep hydrogens as is, add the missing hydrogens, remove all hydrogens, or
            have only the polar ones, see :class:`Mol`.
        **kwargs
            Passed on to :class:`Mol`, for example ``flexible=True``.

        Returns
        -------
        Mol

        See Also
        --------
        from_sdf : Load a molecule with its coordinates.
        from_rdkit : Create a molecule from an RDKit molecule.

        Examples
        --------
        >>> ethanol = Mol.from_smiles("CCO")
        >>> acetate = Mol.from_smiles("CC(=O)[O-]", flexible=True)
        """
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        return cls(mol, hydrogens=hydrogens, **kwargs)

    @classmethod
    def from_rdkit(
        cls,
        mol: Chem.Mol,
        *,
        hydrogens: Literal["keep", "add", "remove", "polar"] = "polar",
        **kwargs,
    ):
        """Create an instance of :class:`Mol` from an RDKit :class:`rdkit.Chem.rdchem.Mol` object.

        The same as ``Mol(mol)``. The RDKit molecule is copied, so it can be changed afterwards
        without affecting the :class:`Mol`. A molecule without a conformer gets one from RDKit's
        ETKDG.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            The input molecule as an RDKit :class:`rdkit.Chem.rdchem.Mol` object.
        hydrogens : {'keep', 'add', 'remove', 'polar'}, default 'polar'
            Whether to keep hydrogens as is, add the missing hydrogens, remove all hydrogens, or
            have only the polar ones, see :class:`Mol`.
        **kwargs
            Passed on to :class:`Mol`, for example ``flexible=True``.

        Returns
        -------
        Mol

        See Also
        --------
        to_rdkit : Get an RDKit copy back.

        Examples
        --------
        >>> from rdkit import Chem
        >>> rd = Chem.MolFromMolFile("ligand.sdf", removeHs=False)
        >>> ligand = Mol.from_rdkit(rd, flexible=True)
        """
        return cls(mol, hydrogens=hydrogens, **kwargs)

    @classmethod
    def from_pdb(
        cls,
        pdb_file: str,
        *,
        hydrogens: Literal["keep", "add", "remove", "polar"] = "polar",
        template_smiles: str = None,
        template_sdf: str = None,
        **kwargs,
    ):
        r"""Create an instance of :class:`Mol` from a PDB file.

        Always sanitizes if template included. A PDB file has no charges: when it has hydrogens,
        an oxygen with one bond and no hydrogen is charged -1 (a carboxylate), and a nitrogen with
        four bonds +1.

        .. warning::
            When loading small molecules from PDB, always include a template.
            Molecules without templates will not be sanitized, and can thus not be used
            in certain scoring functions.


        Parameters
        ----------
        pdb_file : str
            Path to the PDB file containing the molecule.
        hydrogens : {'keep', 'add', 'remove', 'polar'}, default 'polar'
            Whether to keep hydrogens as is, add the missing hydrogens, remove all hydrogens, or
            have only the polar ones, see :class:`Mol`.
        template_smiles : str, optional
            SMILES string representing a reference molecule.
            To sanitize the molecule, either `template_smiles` or
            `template_sdf` must be provided.
        template_sdf : str, optional
            Path to an SDF file containing a reference molecule.
            To sanitize the molecule, either `template_smiles` or
            `template_sdf` must be provided.
        **kwargs
            Passed on to :class:`Mol`, for example ``flexible=True``.

        Returns
        -------
        Mol

        See Also
        --------
        pyrite.io.fix_receptor_pdb : Repair a receptor PDB file before loading it.
        from_sdf : Load a ligand with its bond orders.

        Examples
        --------
        >>> receptor = Mol.from_pdb("receptor.pdb")

        A ligand from a PDB file needs its bond orders from a template:

        >>> ligand = Mol.from_pdb("ligand.pdb", template_smiles="CC(=O)Nc1ccc(O)cc1", flexible=True)
        """
        # Hydrogens are always read: whether the file has them decides the charges below. The
        # constructor removes them for hydrogens="remove".
        mol = Chem.MolFromPDBFile(pdb_file, sanitize=False, removeHs=False)

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

        if not (template_smiles or template_sdf):
            _charge_deprotonated_oxygens(mol)

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
        *,
        hydrogens: Literal["keep", "add", "remove", "polar"] = "polar",
        **kwargs,
    ):
        """Create an instance of :class:`Mol` from an SDF file.

        Only the first record of the file is read. The hydrogens and charges are taken as the file
        has them; whether a hydroxyl or a carboxylate, the bond orders in the file decide.

        Parameters
        ----------
        mol_file : str
            Path to the SDF file containing the molecule.
        hydrogens : {'keep', 'add', 'remove', 'polar'}, default 'polar'
            Whether to keep hydrogens as is, add the missing hydrogens, remove all hydrogens, or
            have only the polar ones, see :class:`Mol`.
        **kwargs
            Passed on to :class:`Mol`, for example ``flexible=True``.

        Returns
        -------
        Mol

        See Also
        --------
        poses_from_sdf : Read all records of an SDF file as poses of a molecule.
        from_pdb : Load a molecule from a PDB file.

        Examples
        --------
        >>> ligand = Mol.from_sdf("ligand.sdf", flexible=True)

        Keep every hydrogen, for example for :class:`~pyrite.scoring.InternalEnergy`:

        >>> ligand = Mol.from_sdf("ligand.sdf", flexible=True, hydrogens="keep")
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

    def _fix_mol_valence(self, sanitize=True, assign_charges=True):
        """Sanitize the molecule, and set the formal charges its valences imply.

        Sanitizes all but the properties, and, unless `assign_charges` is false, sets the formal
        charges that the valences of the input imply (an ammonium).

        Oxygens are left as given: a singly bonded oxygen is a hydroxyl in a SMILES string, an SDF
        or a MOL2 file (their hydrogens are implicit). Only a PDB file says nothing about charges;
        :meth:`from_pdb` charges its deprotonated oxygens itself.

        The charge rules read the valences of the molecule *as it is given*, so they are applied
        once, to the input, and not again after hydrogens were added or removed: an oxygen that
        has just lost its hydrogen would look like an oxide.
        """
        Chem.SanitizeMol(
            self._rdkit,
            sanitizeOps=(Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES),
        )

        for atom in self._rdkit.GetAtoms() if assign_charges else ():
            # print(
            #     atom.GetSymbol(),
            #     atom.GetValence(Chem.ValenceType.EXPLICIT),
            #     atom.GetFormalCharge(),
            # )

            # a neutral nitrogen cannot have four bonds: an ammonium (Lys NZ, His, Arg in a PDB
            # file with hydrogens, or a quaternary amine drawn without its charge)
            if atom.GetSymbol() == "N" and atom.GetValence(Chem.ValenceType.EXPLICIT) == 4:
                atom.SetFormalCharge(+1)
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
        """Find the rotatable torsions of the molecule, and store them for later use.

        This includes identifying the rotatable bonds, the four atoms that define every torsion
        (oriented so that the side with the center atom stays fixed), the atoms every torsion
        moves, and the torsion values of the reference geometry.
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
        # The torsion values of the reference geometry: a pose turns every torsion by its target
        # minus this (see `_pose_positions_kernel`). They depend on which side moves, so they are
        # computed again when the center atom changes.
        self.__torsion_offsets = self.__torsions_in(self._reference_positions)

    @property
    def rdkit(self) -> Chem.Mol:
        """The underlying :class:`rdkit.Chem.rdchem.Mol`, for any RDKit functionality.

        This is the molecule Pyrite works on, not a copy: treat it as read-only (descriptors,
        substructure matches, force fields, ...). To change the molecule, edit a copy
        (:meth:`to_rdkit`) and build a new :class:`Mol` from it. For what RDKit can do, see the
        `RDKit documentation <https://www.rdkit.org/docs/>`_, in particular
        `Getting Started with the RDKit in Python
        <https://www.rdkit.org/docs/GettingStartedInPython.html>`_.

        Returns
        -------
        rdkit.Chem.rdchem.Mol
            The RDKit molecule (not a Pyrite :class:`Mol`).

        See Also
        --------
        to_rdkit : A copy to edit, optionally with a pose as its conformer.

        Examples
        --------
        >>> from rdkit.Chem import Descriptors
        >>> Descriptors.MolWt(ligand.rdkit)
        """
        return self._rdkit

    def to_rdkit(self, pose: Pose | NDArray | None = None) -> Chem.Mol:
        """Return a copy of the underlying :class:`rdkit.Chem.rdchem.Mol`, free to use and to edit.

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
            The copy, with the conformers of this molecule, or one conformer for `pose`.

        Raises
        ------
        ValueError
            When `pose` is a batch.

        See Also
        --------
        rdkit : The molecule itself, read-only.
        pose_to_positions : Only the positions of a pose, without RDKit.

        Examples
        --------
        >>> from rdkit import Chem
        >>> posed = ligand.to_rdkit(pose)
        >>> Chem.MolToMolFile(posed, "pose.mol")
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
        """The number of atoms in the molecule.

        All atoms, the hydrogens it has included. Scoring functions only take the atoms in
        ``scoring_mask`` into account.

        See Also
        --------
        atoms : The atoms themselves.

        Examples
        --------
        >>> ligand = Mol.from_sdf("ligand.sdf", flexible=True)
        >>> ligand.n_atoms
        """
        return self._rdkit.GetNumAtoms()

    @property
    def atoms(self):
        """The :class:`rdkit.Chem.rdchem.Atom` objects of the molecule, in index order.

        The atoms of the RDKit molecule (:attr:`rdkit`): treat them as read-only.

        See Also
        --------
        atom_types : The Pyrite atom type of every atom.

        Examples
        --------
        >>> symbols = [atom.GetSymbol() for atom in ligand.atoms]
        """
        return self._rdkit.GetAtoms()

    @property
    def n_conformers(self) -> int:
        """The number of conformers: the global one, and any made by :meth:`pose_to_conformer`.

        Scoring never adds conformers: positions come from :meth:`pose_to_positions`. Only
        :meth:`pose_to_conformer` with ``new_conf=True`` does, for export or display.

        See Also
        --------
        remove_conformer : Remove a conformer again.

        Examples
        --------
        >>> ligand.n_conformers
        1
        >>> conf_id = ligand.pose_to_conformer(pose, new_conf=True)
        >>> ligand.n_conformers
        2
        """
        return self._rdkit.GetNumConformers()

    def remove_conformer(self, conf_id: int) -> None:
        """Remove a conformer, e.g. one made by :meth:`pose_to_conformer` with ``new_conf=True``.

        The global conformer (the input geometry) should not be removed: the poses are defined
        relative to it.

        Parameters
        ----------
        conf_id : int
            The id of the conformer to remove.

        See Also
        --------
        pose_to_conformer : Put a pose on a (new) conformer.

        Examples
        --------
        >>> conf_id = ligand.pose_to_conformer(pose, new_conf=True)
        >>> ligand.remove_conformer(conf_id)
        """
        self._rdkit.RemoveConformer(conf_id)

    @property
    def atom_types(self):
        """The atom types of all atoms in the molecule.

        The types follow AutoDock Vina's XS types: they decide which atoms are hydrophobic, donors
        or acceptors in the scoring functions.

        Returns
        -------
        numpy.ndarray

        See Also
        --------
        pyrite.AtomType : The atom types.
        atoms : The atoms themselves.

        Examples
        --------
        >>> from pyrite import AtomType
        >>> [AtomType(t).name for t in ligand.atom_types[:3]]
        """
        return self._atom_types

    # endregion

    # region Pose & geometry

    @property
    def positions(self):
        """The positions of all atoms in the global conformer.

        The same as ``get_positions()``. The global conformer is the input geometry, unless
        :meth:`pose_to_conformer` moved it.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_atoms, 3)``.

        See Also
        --------
        get_positions : The positions of any conformer, or of poses.
        pose_to_positions : The positions of poses.

        Examples
        --------
        >>> ligand.positions.shape  # (n_atoms, 3)
        """
        return self._rdkit.GetConformer().GetPositions()

    def get_positions(self, conf_id: int = -1, poses=None) -> NDArray[np.float32]:
        """Return the positions of all atoms in a specific conformer.

        To get the positions of poses, prefer :meth:`pose_to_positions`: it does not need a
        conformer.

        Parameters
        ----------
        conf_id : int, default -1
            The conformer id to retrieve positions from. By default selects the global conformer.
        poses : Pose or Poses, optional
            If given, the positions of these poses instead, as :meth:`pose_to_positions`.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_atoms, 3)``, or ``(n, n_atoms, 3)`` for a batch of poses.

        See Also
        --------
        pose_to_positions : The positions of poses.

        Examples
        --------
        >>> ligand.get_positions().shape  # (n_atoms, 3)
        >>> ligand.get_positions(poses=poses).shape  # (n_poses, n_atoms, 3)
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

        See Also
        --------
        pose_to_conformer : Put a pose on a conformer, for export and display.
        to_rdkit : An RDKit copy with a pose as its conformer.
        pose_from_positions : The reverse: a pose from positions.

        Examples
        --------
        >>> ligand.pose_to_positions(ligand.input_pose).shape  # (n_atoms, 3)
        >>> ligand.pose_to_positions(poses).shape  # (n_poses, n_atoms, 3)
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

        reference = self._reference_positions
        positions = _pose_positions_kernel(
            np.ascontiguousarray(values, dtype=np.float64),
            reference - reference[self._center_atom],
            layout.rot_type == "euler",
            layout.rot_dim,
            *self.__packed_torsions,
            self.__torsion_offsets,
        )
        return positions[0] if single else positions

    def pose_gradient(self, pose, positions: NDArray, forces: NDArray) -> NDArray:
        """Turn the gradient of a score per atom into its gradient per pose variable.

        For a scoring function with an analytic gradient: compute ``dS/dx`` for every atom (how
        the score changes when the atom moves), and this gives ``dS/dpose``, the gradient a local
        optimizer needs. Every pose variable moves the atoms rigidly, so its derivative is a sum
        over the atoms it moves: the forces for a translation, their torque for a rotation or a
        torsion.

        Parameters
        ----------
        pose : Pose or array_like
            The pose, shape ``(n_dims,)``, in the layout of this molecule.
        positions : array_like
            The atom positions of `pose`, shape ``(n_atoms, 3)`` (from :meth:`pose_to_positions`).
        forces : array_like
            ``dS/dx`` for every atom, shape ``(n_atoms, 3)``; zero for atoms the score ignores.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_dims,)``, in the layout of this molecule.

        See Also
        --------
        pyrite.scoring.ScoringFunction.get_score_and_gradient : The score of a pose and its
            gradient.
        pose_to_positions : The positions of a pose.

        Examples
        --------
        In a scoring function:

        >>> def _score_and_gradient(self, pose, computed):
        ...     positions = computed[self._position_dep]
        ...     score, forces = ...  # dS/dx per atom
        ...     return score, self.mol.pose_gradient(pose, positions, forces)
        """
        layout = self.layout
        return _pose_gradient_kernel(
            np.ascontiguousarray(positions, dtype=np.float64),
            np.ascontiguousarray(forces, dtype=np.float64),
            np.asarray(pose, dtype=np.float64),
            layout.rot_type == "euler",
            layout.rot_dim,
            self._center_atom,
            *self.__packed_torsions[:3],
        )

    @property
    def rotatable_torsions(self):
        """The rotatable torsions of the molecule.

        Which side of a torsion moves depends on the center atom: the side with the center atom
        stays fixed. Setting :attr:`center_atom` orients them again.

        Returns
        -------
        list
            A list containing all rotatable torsions in the molecule,
            indicated by four atom indices. An entry looks like
            ``[i, j, k, l]``, where the rotated bond is between
            ``j`` and ``k``, and all atoms attached to ``k`` are moved.

        See Also
        --------
        torsions : The values of the torsions.
        n_tors : The number of torsions.

        Examples
        --------
        >>> a, b, c, d = ligand.rotatable_torsions[0]  # the bond b-c turns the atoms beyond c
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

        See Also
        --------
        input_pose : The pose of the input geometry, with its torsions.
        rotatable_torsions : The atoms that define every torsion.

        Examples
        --------
        >>> ligand.torsions.shape  # (n_tors,)
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
    def layout(self) -> PoseLayout:
        """The layout of the poses of this molecule.

        It follows from the molecule: the rotation type it was made with, and its number of
        torsions. Every pose and scoring function of this molecule uses it.

        See Also
        --------
        PoseLayout : How the variables of a pose are laid out.
        n_tors : The number of torsions.

        Examples
        --------
        >>> ligand = Mol.from_sdf("ligand.sdf", flexible=True)
        >>> ligand.layout  # PoseLayout(rot_type='euler', n_tors=...)
        """
        return self._layout

    @property
    def scoring_mask(self) -> NDArray[np.bool_]:
        """Which atoms scoring functions take into account, one boolean per atom.

        By default every atom but the hydrogens (``ignore_hydrogens=True``): polar hydrogens still
        decide which atoms are donors, but are not scored themselves. It is computed once, so
        every scoring function of this molecule agrees on it.

        See Also
        --------
        atom_types : The atom type of every atom.

        Examples
        --------
        >>> ligand.scoring_mask.sum()  # the scored atoms
        """
        return self._scoring_mask

    @property
    def n_tors(self):
        """The number of torsions in the molecule.

        Zero for a molecule that is not ``flexible``. Every torsion is one variable of a pose, after
        the rotation and the translation.

        See Also
        --------
        rotatable_torsions : The atoms that define every torsion.
        PoseLayout.n_dims : The number of variables of a pose.

        Examples
        --------
        >>> ligand = Mol.from_sdf("ligand.sdf", flexible=True)
        >>> ligand.n_tors
        """
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

        See Also
        --------
        pose_to_positions : The positions of a pose, without a conformer.
        to_sdf : Write poses to an SDF file.

        Examples
        --------
        >>> conf_id = ligand.pose_to_conformer(pose, new_conf=True)
        >>> ligand.viewer
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

        Rotations are about the center atom, the translation of a pose is its position, and it is on
        the fixed side of every torsion. Setting it orients the torsions again and puts the global
        conformer back in its input orientation; existing poses keep their meaning.

        Returns
        -------
        int

        See Also
        --------
        position : The position of the center atom.
        rotatable_torsions : The torsions, oriented from the center atom.

        Examples
        --------
        >>> ligand.center_atom
        >>> ligand.center_atom = 10  # another atom
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

        See Also
        --------
        center_atom : The index of the center atom.
        Pose : The translation of a pose is the position of the center atom.

        Examples
        --------
        >>> x, y, z = ligand.position
        """
        center_atom_coords = self._rdkit.GetConformer().GetAtomPosition(self._center_atom)
        return [center_atom_coords.x, center_atom_coords.y, center_atom_coords.z]

    @property
    def rotation(self):
        """The rotation of the global conformer.

        The rotation of the last pose put on the global conformer with :meth:`pose_to_conformer`,
        in the representation of the layout; no rotation for the input geometry.

        Returns
        -------
        numpy.ndarray
            Shape ``(rot_dim,)``.

        See Also
        --------
        pose_to_conformer : Put a pose on the global conformer.
        PoseLayout.identity_rotation : No rotation.

        Examples
        --------
        >>> ligand.rotation
        array([0., 0., 0.])
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

        See Also
        --------
        PoseLayout.sample_random_torsions : Independent random torsions.
        pyrite.search.place_in : Place a molecule in a pocket with these torsions.

        Examples
        --------
        >>> torsions = ligand.get_n_conformer_torsion_configurations(10)
        >>> torsions.shape  # (10, n_tors)
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

        See Also
        --------
        poses_from_sdf : Read the poses back.
        pose_to_conformer : Put a pose on a conformer.

        Examples
        --------
        >>> ligand.to_sdf("docked.sdf", poses=best_poses)
        >>> ligand.to_sdf("input.sdf")
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

    @property
    def input_pose(self) -> Pose:
        """The pose of the geometry this molecule was built from.

        The pose that puts every atom where the input had it: no rotation, the center atom at its
        input position, the input torsions. For a crystal ligand this is the crystal pose, to
        score, to start a search from, or to compare docked poses with. It does not change when
        the global conformer is moved (:meth:`pose_to_conformer`).

        Returns
        -------
        Pose
            In the layout of this molecule; ``pose_to_positions(mol.input_pose)`` gives the input
            coordinates.

        See Also
        --------
        pose_from_positions : A pose from any coordinates of the molecule.
        pyrite.scoring.RMSD : The RMSD of a pose to the input geometry.

        Examples
        --------
        >>> crystal_score = scoring_function.get_score(ligand.input_pose)
        >>> result = basin_hopping.run(ligand.input_pose, niter=50)
        """
        layout = self.layout
        reference = self._reference_positions
        values = np.concatenate(
            [layout.identity_rotation, reference[self._center_atom], self.__torsions_in(reference)]
        )
        return Pose(values, layout)

    def pose_from_positions(self, positions: NDArray, rmsd_delta: float = 0.5) -> Pose:
        """Make a pose of this molecule from coordinates of its atoms.

        The torsions are measured in `positions`, and the rotation and translation are the rigid
        fit (Kabsch) of this molecule's geometry with those torsions onto them. Useful for a
        conformer from another program or file, of the same molecule with its atoms in the same
        order.

        Parameters
        ----------
        positions : array_like
            Shape ``(n_atoms, 3)``, in the atom order of this molecule.
        rmsd_delta : float, default 0.5
            The largest RMSD (in Angstrom) allowed between `positions` and the pose made of them.
            Bond lengths and angles are not part of a pose, so coordinates that differ in those
            cannot be reproduced exactly; this is how much is accepted.

        Returns
        -------
        Pose
            In the layout of this molecule.

        Raises
        ------
        ValueError
            When `positions` has the wrong shape, is not finite, or cannot be reproduced within
            `rmsd_delta`.

        See Also
        --------
        poses_from_sdf : Poses from the conformers of an SDF file.
        pose_to_positions : The reverse: the positions of a pose.

        Examples
        --------
        >>> from rdkit import Chem
        >>> other = Chem.MolFromMolFile("other_program.sdf", removeHs=False)
        >>> pose = ligand.pose_from_positions(other.GetConformer().GetPositions())
        """
        positions = np.asarray(positions, dtype=float)
        if positions.shape != (self.n_atoms, 3):
            raise ValueError(
                f"positions must have shape ({self.n_atoms}, 3), got {positions.shape}."
            )
        if not np.isfinite(positions).all():
            raise ValueError("positions must be finite.")
        rmsd, values = self.__fit_pose(positions)
        if rmsd > rmsd_delta:
            raise ValueError(
                f"The positions cannot be reproduced as a pose: RMSD {rmsd:.2f} > rmsd_delta "
                f"{rmsd_delta}."
            )
        return Pose(values, self.layout)

    def __torsions_in(self, positions: NDArray) -> NDArray:
        """The rotatable torsion angles of this molecule in `positions` (radians)."""
        work = Chem.Mol(self._rdkit, False, self._rdkit.GetConformer().GetId())
        work.GetConformer().SetPositions(np.asarray(positions, dtype=float))
        return np.array(
            [
                Chem.rdMolTransforms.GetDihedralRad(work.GetConformer(), *torsion)
                for torsion in self.__rotatable_torsions
            ],
            dtype=float,
        )

    def __fit_pose(self, observed: NDArray) -> tuple[float, NDArray]:
        """The pose values that best reproduce `observed`, and the RMSD they leave."""
        layout = self.layout
        torsions = self.__torsions_in(observed)
        # the reference geometry with these torsions, the center atom at the origin
        identity = np.concatenate([layout.identity_rotation, np.zeros(3), torsions])
        shape = self.pose_to_positions(identity)

        # the rigid motion that takes it onto the observed positions (Kabsch)
        shape_mean, observed_mean = shape.mean(axis=0), observed.mean(axis=0)
        u, _, vt = np.linalg.svd((shape - shape_mean).T @ (observed - observed_mean))
        reflection = np.sign(np.linalg.det(vt.T @ u.T))
        rotation = vt.T @ np.diag([1.0, 1.0, reflection]) @ u.T
        translation = observed_mean - rotation @ shape_mean

        rmsd = np.sqrt(np.mean(np.sum((shape @ rotation.T + translation - observed) ** 2, 1)))
        values = np.concatenate([layout.rotation_from_matrix(rotation), translation, torsions])
        return float(rmsd), values

    def poses_from_sdf(self, file: str, rmsd_delta: float = 0.5) -> Poses:
        """Read the conformers in an SDF file as poses of this molecule.

        Every conformer becomes a pose as in :meth:`pose_from_positions`. The conformers must be
        of this molecule, with the same atoms in the same order.

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

        See Also
        --------
        to_sdf : Write poses to an SDF file.
        pose_from_positions : A pose from coordinates.

        Examples
        --------
        >>> poses = ligand.poses_from_sdf("docked.sdf")
        >>> scores = scoring_function.batch_scores(poses)
        """
        layout = self.layout
        template = self._rdkit
        bonds = {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in template.GetBonds()}
        elements = [a.GetAtomicNum() for a in template.GetAtoms()]

        values = []
        for record in self.__read_sdf_records(file):
            same_molecule = [a.GetAtomicNum() for a in record.GetAtoms()] == elements and {
                frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in record.GetBonds()
            } == bonds
            if not same_molecule:
                raise ValueError(
                    f"Record {len(values)} of {file} is not this molecule (or its atoms are in a "
                    "different order)."
                )
            rmsd, pose_values = self.__fit_pose(record.GetConformer().GetPositions())
            if rmsd > rmsd_delta:
                raise ValueError(
                    f"Record {len(values)} of {file} cannot be reproduced as a pose: RMSD "
                    f"{rmsd:.2f} > rmsd_delta {rmsd_delta}."
                )
            values.append(pose_values)

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

        The options are used by :attr:`png`, :attr:`svg`, :attr:`viewer` and
        :class:`~pyrite.Viewer`.

        Parameters
        ----------
        options : dict
            The options to apply, on top of the current ones. For a ligand: ``size`` (width and
            height in pixels), ``colorPalette``, ``colorscheme``, ``highlight`` and ``note``; for a
            protein also ``color``, ``style``, ``surfacetype``, ``surfacecolor``,
            ``surfaceopacity``, ``stickresidues`` (residue numbers, or names such as ``"HIS"``, to show
            as sticks) and ``hideprotein``. A ``surfacetype`` of None leaves out the surface.

        See Also
        --------
        viewer : Show the molecule in 3D.
        pyrite.Viewer : Show several objects together.

        Examples
        --------
        >>> ligand.set_draw_options({"size": (600, 400), "note": "type"})
        >>> receptor.set_draw_options({"surfaceopacity": 0.5, "stickresidues": [57, 102]})
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
            # A residue number, or a residue name
            key = "resn" if isinstance(res, str) else "resi"
            viewer.view.setStyle(
                {"model": m_id, key: res, "byres": True},
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

    def _repr_html_(self):
        return self.viewer._repr_html_()

    @property
    def png(self):
        """A png of this molecule for use in jupyter notebooks.

        A 2D drawing, of the size in the draw options.

        Returns
        -------
        ~IPython.display.Image

        See Also
        --------
        svg : The same as svg.
        set_draw_options : Set the size.

        Examples
        --------
        >>> ligand.png
        """
        return Image(self._repr_png_(), embed=True)

    @property
    def svg(self):
        """A svg of this molecule for use in jupyter notebooks.

        A 2D drawing, of the size in the draw options.

        Returns
        -------
        ~IPython.display.SVG

        See Also
        --------
        png : The same as png.
        set_draw_options : Set the size.

        Examples
        --------
        >>> ligand.svg
        """
        return SVG(self._repr_svg_())

    @property
    def viewer(self):
        """A viewer containing this molecule.

        A 3D view of the global conformer, with the draw options of this molecule.

        Returns
        -------
        Viewer

        See Also
        --------
        pyrite.Viewer : Show several objects together.
        set_draw_options : Set how the molecule is drawn.

        Examples
        --------
        >>> ligand.viewer
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

        # Draw a copy, laid out in 2D (a 3D conformer drawn flat is hard to read); the atom notes
        # are properties on its atoms.
        mol = Chem.Mol(self._rdkit)
        rdDepictor.Compute2DCoords(mol)
        match self.draw_options.get("note", "").lower():
            case "idx":
                for a in mol.GetAtoms():
                    a.SetProp("atomNote", f"{a.GetIdx()}")
            case "type":
                for a in mol.GetAtoms():
                    a.SetProp("atomNote", f"{str(self._atom_types[a.GetIdx()])}")

        highlight = self.draw_options.get("highlight") or []
        if isinstance(highlight, str):
            highlight = [self._center_atom] if highlight.lower() == "center" else []

        d2d.DrawMolecule(mol, highlightAtoms=[int(i) for i in highlight])
        d2d.FinishDrawing()
        return d2d.GetDrawingText()

    # endregion

    # region Copying

    def copy(self) -> Mol:
        """Return a full, independent copy of this molecule.

        The copy has the RDKit molecule with all its conformers, and every derived value, as they
        are: unlike ``Mol(mol.rdkit)``, nothing is computed again.

        Returns
        -------
        Mol
            The copy.

        See Also
        --------
        to_rdkit : An RDKit copy of the molecule.

        Examples
        --------
        >>> other = ligand.copy()
        >>> other.center_atom = 0  # does not change ligand
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
    """
    Describes how the variables of a pose are laid out.

    A pose is a single vector ``[rotation, translation, torsions]``. The layout holds how many
    variables every part has, and where they are in the vector.

    Parameters
    ----------
    rot_type : {'euler', 'quat'}
        The representation of the rotation: Euler angles ``(roll, pitch, yaw)``, or a quaternion
        ``(w, x, y, z)``.
    n_tors : int
        The number of torsions.

    Attributes
    ----------
    rot_dim : int
        The number of rotation variables: 3 for Euler angles, 4 for a quaternion.
    rot_slice, trans_slice, tors_slice : slice
        Where the rotation, translation and torsions are in a pose vector.
    n_dims : int
        The number of variables of a pose: ``rot_dim + 3 + n_tors``.
    identity_rotation : numpy.ndarray
        The rotation that does not rotate, in this layout's representation.

    Raises
    ------
    ValueError
        If `rot_type` is not ``'euler'`` or ``'quat'``.

    See Also
    --------
    Pose : A pose, laid out as described by a layout.
    Mol.layout : The layout of the poses of a molecule.

    Examples
    --------
    >>> layout = PoseLayout("euler", n_tors=6)
    >>> layout.n_dims
    12
    >>> layout.trans_slice
    slice(3, 6, None)
    >>> PoseLayout("quat", n_tors=6).n_dims
    13
    """

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
        """The length of a pose vector in this layout.

        ``rot_dim + 3 + n_tors``: 3 or 4 rotation variables, the translation, and the torsions.

        See Also
        --------
        rot_dim : The number of rotation variables.

        Examples
        --------
        >>> PoseLayout("euler", n_tors=6).n_dims
        12
        """
        return self.rot_dim + 3 + self.n_tors

    @property
    def identity_rotation(self) -> NDArray[np.float32]:
        """The rotation that does not rotate, in this layout's representation.

        Zeros for Euler angles, ``(1, 0, 0, 0)`` for a quaternion.

        See Also
        --------
        compose_rotation : Rotate a rotation.

        Examples
        --------
        >>> PoseLayout("euler", 0).identity_rotation
        array([0., 0., 0.])
        >>> PoseLayout("quat", 0).identity_rotation
        array([1., 0., 0., 0.], dtype=float32)
        """
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

        See Also
        --------
        pyrite.search.random_hop : The hop of BasinHopping, which uses this.

        Examples
        --------
        >>> import numpy as np
        >>> small_turn = np.array([0.0, 0.0, 0.1])  # 0.1 radian about the z axis
        >>> new_rotation = layout.compose_rotation(pose.rotation, small_turn)
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

        For Euler angles ``(roll, pitch, yaw)``, ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``; a
        quaternion is normalised first.

        Parameters
        ----------
        rotation : ndarray
            Shape ``(rot_dim,)`` or ``(n, rot_dim)``.

        Returns
        -------
        ndarray
            Shape ``(3, 3)`` or ``(n, 3, 3)``. It acts on column vectors: ``x' = R @ x``.

        See Also
        --------
        rotation_from_matrix : The reverse.

        Examples
        --------
        >>> layout.rotation_matrix(layout.identity_rotation)
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])
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

        See Also
        --------
        rotation_matrix : The reverse.

        Examples
        --------
        >>> rotation = layout.rotation_from_matrix(np.eye(3))
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

        See Also
        --------
        sample_random_torsions : Random torsions.
        Poses.from_parts : Assemble poses.

        Examples
        --------
        >>> rotations = layout.sample_random_rotations(100, rng=np.random.default_rng(0))
        >>> rotations.shape
        (100, 3)
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

        See Also
        --------
        Mol.get_n_conformer_torsion_configurations : Realistic torsions, from conformers.

        Examples
        --------
        >>> torsions = layout.sample_random_torsions(100, rng=np.random.default_rng(0))
        >>> torsions.shape
        (100, 6)
        """
        rng = np.random.default_rng() if rng is None else rng
        return rng.uniform(-np.pi, np.pi, size=(n, self.n_tors))


class Pose:
    """
    Represents a single pose of a molecule.

    A pose is a vector ``[rotation, translation, torsions]``, laid out as described by `layout`.
    The translation is the position of the center atom of the :class:`Mol`. A pose can be used
    wherever a numpy array is expected.

    Parameters
    ----------
    v : array_like
        The pose vector, of shape ``(n_dims,)``. It is not copied.
    layout : PoseLayout
        The layout of `v`.

    Attributes
    ----------
    rotation, translation, torsions : numpy.ndarray
        Views of the parts of the pose vector.

    Raises
    ------
    ValueError
        If `v` does not have the shape of the layout.

    See Also
    --------
    Poses : A batch of poses.
    PoseLayout : How the variables of a pose are laid out.
    Mol.pose_to_positions : The atom positions of a pose.

    Examples
    --------
    >>> import numpy as np
    >>> ligand = Mol.from_sdf("ligand.sdf", flexible=True)
    >>> pose = ligand.input_pose
    >>> pose.translation  # the position of the center atom
    >>> shifted = np.asarray(pose).copy()
    >>> shifted[ligand.layout.trans_slice] += [1.0, 0.0, 0.0]  # 1 A along x
    >>> moved = Pose(shifted, ligand.layout)
    """

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
        """Create a pose from a copy of `v`.

        Unlike ``Pose(v, layout)``, the pose does not share its values with `v`.

        Parameters
        ----------
        v : array_like
            The pose vector, of shape ``(n_dims,)``.
        layout : PoseLayout
            The layout of `v`.

        Returns
        -------
        Pose

        See Also
        --------
        Poses.from_array : The same, for a batch.

        Examples
        --------
        >>> pose = Pose.from_array(values, ligand.layout)
        """
        return cls(np.asarray(v).copy(), layout)


class Poses:
    """
    Represents a batch of poses of a molecule.

    The poses are the rows of an array of shape ``(n, n_dims)``, laid out as described by
    `layout`. Indexing with an integer gives a :class:`Pose`, with a slice or an index array a
    :class:`Poses`. A batch of poses can be used wherever a numpy array is expected.

    Parameters
    ----------
    vs : array_like
        The pose vectors, of shape ``(n, n_dims)``. They are not copied.
    layout : PoseLayout
        The layout of the poses.

    Attributes
    ----------
    rotation, translation, torsions : numpy.ndarray
        Views of the parts of the pose vectors, every one with ``n`` rows.

    Raises
    ------
    ValueError
        If `vs` does not have the shape of the layout.

    See Also
    --------
    Pose : A single pose.
    pyrite.scoring.ScoringFunction.batch_scores : Score many poses at once.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> layout = ligand.layout
    >>> poses = Poses.from_parts(
    ...     layout.sample_random_rotations(10, rng),
    ...     rng.normal(ligand.position, 1.0, (10, 3)),
    ...     ligand.get_n_conformer_torsion_configurations(10),
    ...     layout=layout,
    ... )
    >>> len(poses), poses[0].translation.shape
    (10, (3,))
    >>> scores = scoring_function.batch_scores(poses)
    """

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
        """Create poses from a copy of `vs`.

        Unlike ``Poses(vs, layout)``, the poses do not share their values with `vs`.

        Parameters
        ----------
        vs : array_like
            The pose vectors, of shape ``(n, n_dims)``.
        layout : PoseLayout
            The layout of the poses.

        Returns
        -------
        Poses

        See Also
        --------
        Poses.from_parts : Assemble poses from their parts.
        Poses.from_list : Combine single poses.

        Examples
        --------
        >>> poses = Poses.from_array(values, ligand.layout)
        """
        return cls(np.asarray(vs).copy(), layout)

    @classmethod
    def from_parts(
        cls,
        rotation: NDArray,
        translation: NDArray,
        torsions: NDArray | None = None,
        layout: PoseLayout | None = None,
    ):
        """Assemble poses from their rotations, translations and torsions.

        The usual way to make poses for a search: random rotations from the layout, positions in the
        binding site, and torsions from conformers. The layout follows from the parts: 3 rotation
        variables are Euler angles, 4 a quaternion, and the number of torsions is the number of
        columns of `torsions`.

        Parameters
        ----------
        rotation : array_like
            Shape ``(n, 3)`` for Euler angles, or ``(n, 4)`` for quaternions.
        translation : array_like
            Shape ``(n, 3)``.
        torsions : array_like, optional
            Shape ``(n, n_tors)``. By default no torsions, for a rigid molecule.
        layout : PoseLayout, optional
            The layout of the poses. By default inferred from the parts; if given, the parts must
            fit it, for example ``mol.layout`` to make sure the poses fit a molecule.

        Returns
        -------
        Poses

        Raises
        ------
        ValueError
            If the parts do not have the same number of poses, `rotation` does not have 3 or 4
            columns, or the parts do not fit `layout`.

        See Also
        --------
        PoseLayout.sample_random_rotations : Uniformly random rotations.
        Mol.get_n_conformer_torsion_configurations : Realistic torsions.
        pyrite.search.place_in : Place a molecule in a pocket.

        Examples
        --------
        >>> poses = Poses.from_parts(rotations, translations, torsions)
        >>> poses = Poses.from_parts(rotations, translations, torsions, layout=ligand.layout)
        >>> rigid = Poses.from_parts(rotations, translations)
        """
        rotation = np.atleast_2d(np.asarray(rotation, dtype=float))
        translation = np.atleast_2d(np.asarray(translation, dtype=float))
        n = len(translation)
        torsions = np.empty((n, 0)) if torsions is None else np.asarray(torsions, dtype=float)
        torsions = torsions.reshape(n, -1) if torsions.ndim < 2 else torsions  # one torsion
        if not len(rotation) == n == len(torsions):
            raise ValueError(
                f"The parts have {len(rotation)}, {n} and {len(torsions)} poses: they must agree."
            )
        if layout is None:
            rot_type = {3: "euler", 4: "quat"}.get(rotation.shape[1])
            if rot_type is None:
                raise ValueError(
                    f"A rotation has 3 (Euler angles) or 4 (a quaternion) variables, "
                    f"got {rotation.shape[1]}."
                )
            layout = PoseLayout(rot_type, torsions.shape[1])
        elif rotation.shape[1] != layout.rot_dim or torsions.shape[1] != layout.n_tors:
            raise ValueError(
                f"The parts do not fit {layout}: {rotation.shape[1]} rotation variables and "
                f"{torsions.shape[1]} torsions."
            )
        vs = np.empty((n, layout.n_dims))
        vs[:, layout.rot_slice] = rotation
        vs[:, layout.trans_slice] = translation
        vs[:, layout.tors_slice] = torsions
        return cls(vs, layout)

    @classmethod
    def from_list(cls, poses: list[Pose], layout: PoseLayout | None = None):
        """Create a batch from a list of poses.

        For example to collect the results of separate searches into one batch.

        Parameters
        ----------
        poses : list of Pose
            The poses. They are copied into one array.
        layout : PoseLayout, optional
            The layout of the poses. Defaults to the layout of the first pose.

        Returns
        -------
        Poses

        See Also
        --------
        Poses.from_array : Create poses from an array.

        Examples
        --------
        >>> results = [basin_hopping.run(pose, niter=50) for pose in starts]
        >>> poses = Poses.from_list([result.x for result in results])
        """
        return cls(np.stack([np.asarray(p) for p in poses]), layout or poses[0].layout)
