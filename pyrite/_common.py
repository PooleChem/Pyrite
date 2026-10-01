from __future__ import annotations

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
    _compose_delta_transform,
    _rotation_matrix_from_euler,
    _rotation_matrix_from_quat,
    _rotation_matrix_to_euler,
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


class Mol(Chem.Mol):
    """
    Representation of a molecule.

    The Mol class provides methods for initializing molecules from various sources,
    such as SMILES strings, PDB files, or objects. It also assigns atom
    types, rotatable torsions, and the center point of the molecule. Furthermore, it allows for
    easy manipulation of ligand position, rotation, and torsion angles.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The :class:`~rdkit.Chem.rdchem.Mol` object representing the molecule.
    center_atom : int, optional
        The index of the atom to use as the center point for rotations.
        By default, the center atom is determined automatically by selecting the heavy atom closest
        to the initial conformer centroid.
    ignore_non_polar_hydrogens : bool, default True
        Whether non-polar hydrogens are excluded from `scoring_mask` — the mask every
        scoring function built on this molecule uses to decide which atoms count.


    """

    # region Construction

    def __new__(cls, mol: Chem.Mol = None, **kwargs):
        if mol is None:
            inst = super().__new__(cls)
        else:
            inst = Chem.Mol(mol)
            inst.__class__ = cls

        return inst

    def __init__(
        self,
        mol: Chem.Mol = None,
        hydrogens: Literal["keep", "add", "remove"] = "remove",
        flex_hydrogens: bool = False,
        flexible: bool = False,  # TODO: allow list of resids. No. Read everything rigid, and allow for auto setting of torsions, or manual, or by resid.
        center_atom: int = None,
        rotation_type: Literal["euler", "quat"] = "euler",
        ignore_non_polar_hydrogens: bool = True,
    ):
        self.__rotatable_torsions = np.array([], dtype=object)
        self.__torsion_angles = np.array([])

        self._fix_mol_valence(sanitize=False)  # TODO: sanitize?

        # TODO: does this even do anything?
        if hydrogens == "add":
            mol = Chem.AllChem.AddHs(mol, addCoords=True)
        elif hydrogens == "remove":
            mol = Chem.AllChem.RemoveHs(mol)

        # The center atom is chosen from the conformer, so a molecule without one (e.g. from a
        # SMILES string) needs it embedded first.
        if self.GetNumConformers() == 0:
            Chem.SanitizeMol(self)
            params = Chem.AllChem.ETKDGv3()
            params.randomSeed = 0xC0FFEE
            if Chem.AllChem.EmbedMolecule(self, params) == -1:  # TODO: cant do if not sanitized.
                raise ValueError("RDKit could not embed a 3D conformer of the molecule.")

        # TODO: this doesnt make sense for proteins. In a protein, all torsions should be oriented wrt the backbone, not the center atom.
        self._center_atom = center_atom if center_atom is not None else self.__get_center_atom()

        # TODO: make property. Setting it will then compute the torsions if needed. Perhaps bool | list ?
        self.is_flexible = flexible
        if self.is_flexible:
            self.__compute_rotatable_torsions(flex_hydrogens)

        self.__assign_atom_types()

        # Which atoms scoring functions should consider — computed once, here, so every
        # scoring function built on this Mol agrees, instead of each one recomputing (and
        # potentially disagreeing about) the same mask from the same atom_types.
        self.scoring_mask = ~(
            ignore_non_polar_hydrogens
            & (
                (self._atom_types == AtomType.Hydrogen)
                | (self._atom_types == AtomType.PolarHydrogen)
            )
        )

        Chem.rdPartialCharges.ComputeGasteigerCharges(self)

        # Set the layout
        self.layout: PoseLayout = PoseLayout(rotation_type, len(self.__rotatable_torsions))

        self.__cur_rotation_matrix = np.eye(3)
        self.__cur_rotation = np.zeros(self.layout.rot_dim)

        # If more than 1000 heavy atoms: assume protein for viewing
        if self.GetNumHeavyAtoms() >= VIEWER_PROTEINS_HEAVY_ATOMS_CUTOFF:
            self.draw_options = self.__default_protein_draw_options.copy()
        else:
            self.draw_options = self.__default_ligand_draw_options.copy()

    @classmethod
    def from_smiles(
        cls,
        smiles: str,
        hydrogens: Literal["keep", "add", "remove"] = "remove",
        **kwargs,
    ):
        """Constructs an instance of :class:`Mol` from a SMILES string representation of a molecule.

        Parameters
        ----------
        smiles : str
            The SMILES string representation of the molecule.
        hydrogens : {'keep', 'add', 'remove'}, default 'remove'
            Whether to keep hydrogens as is, add additional hydrogens, or remove all hydrogens.

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
        hydrogens: Literal["keep", "add", "remove"] = "remove",
        **kwargs,
    ):
        """Create an instance of :class:`Mol` from an RDKit :class:`~rdkit.Chem.rdchem.Mol` object.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            The input molecule as an RDKit :class:`~rdkit.Chem.rdchem.Mol` object.
        hydrogens : {'keep', 'add', 'remove'}, default 'remove'
            Whether to keep hydrogens as is, add additional hydrogens, or remove all hydrogens.

        Returns
        -------
        Mol
        """
        return cls(mol, hydrogens=hydrogens, **kwargs)

    @classmethod
    def from_pdb(
        cls,
        pdb_file: str,
        hydrogens: Literal["keep", "add", "remove"] = "remove",
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
        hydrogens : {'keep', 'add', 'remove'}, default 'remove'
            Whether to keep hydrogens as is, add additional hydrogens, or remove all hydrogens.
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

        if hydrogens == "add":
            mol = Chem.AllChem.AddHs(mol)

        if template_smiles or template_sdf:
            if template_smiles:
                template_mol = Chem.MolFromSmiles(template_smiles)
            else:
                template_mol = Chem.MolFromMolFile(template_sdf)

            mol = Chem.AllChem.AssignBondOrdersFromTemplate(template_mol, mol)

            Chem.AssignStereochemistryFrom3D(mol)
            Chem.SanitizeMol(mol)

        h_for_init = "keep" if hydrogens == "add" else hydrogens
        return cls(mol, hydrogens=h_for_init, **kwargs)

    @classmethod
    def from_sdf(
        cls,
        mol_file: str,
        hydrogens: Literal["keep", "add", "remove"] = "remove",
        **kwargs,
    ):
        """Creates an instance of :class:`Mol` from an SDF file.

        Parameters
        ----------
        mol_file : str
            Path to the SDF file containing the molecule.
        hydrogens : {'keep', 'add', 'remove'}, default 'remove'
            Whether to keep hydrogens as is, add additional hydrogens, or remove all hydrogens.

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

    @classmethod
    def v_from_sdf(
        cls,
        mol_file: str,
        hydrogens: Literal["keep", "add", "remove"] = "remove",
        rmsd_delta: float = 0.5,
        **kwargs,
    ):
        """Creates a :class:`Mol` from an SDF file, and also returns a list of variables
        representing the conformations in the SDF file.

        Parameters
        ----------
        mol_file : str
            Path to the SDF file containing the molecule, with multiple conformations.
            All molecules in the SDF file should be equal.
        hydrogens : {'keep', 'add', 'remove'}, default 'remove'
            Whether to keep hydrogens as is, add additional hydrogens, or remove all hydrogens.
        rmsd_delta : float, default 0.5
            The maximum RMSD between the conformations in the SDF file after alignment. This is
            needed when, for example, the SDF file is generated with a program that modifies bond
            angles and -lengths, such that they are different between conformations. These
            properties are currently not considered as variables, and will thus be lost. The
            molecules will be aligned as closely as possible, with a maximum RMSD of `rmsd_delta`.

        Raises
        ------
        ValueError
            When the molecules in the SDF file are not equal, or can't be aligned within an error of
            `rmsd_delta`.


        Returns
        -------
        Mol
        variables : numpy.ndarray
            The conformations in the SDF file, represented by a tuple of size ``(6 + n_tors)``,
            as expected by e.g. :meth:`update`.
        """
        # RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromMolFile(
            mol_file,
            removeHs=(hydrogens == "remove"),
            strictParsing=False,
            sanitize=False,
        )
        # mol = _fix_mol_valence(mol)
        # mol = Chem.AllChem.AddHs(mol, addCoords=True)
        lig = cls(mol, **kwargs)

        canon_smiles = Chem.CanonSmiles(Chem.MolToSmiles(mol))

        # Alignment atom map
        matches = lig.GetSubstructMatches(lig, uniquify=True, useChirality=True)
        atom_map = [list(zip(range(lig.GetNumAtoms()), match)) for match in matches]

        atom_map = [t for sub in atom_map for t in sub]  # Flatten

        v = []
        with Chem.SDMolSupplier(mol_file, removeHs=(hydrogens == "remove"), sanitize=False) as supl:
            for pose in supl:
                if Chem.CanonSmiles(Chem.MolToSmiles(pose)) != canon_smiles:
                    raise ValueError("Molecules in SDF file are not equal.")
                # TODO: dont create whole ligand every time.
                pose_lig = cls(pose, **kwargs)
                pose_torsions = pose_lig.torsions

                # Set torsions equal for alignment
                pose_lig.set_torsions(lig.torsions)

                # Align molecule
                rmsd, transform = Chem.rdMolAlign.GetAlignmentTransform(
                    lig, pose_lig, atomMap=atom_map
                )
                if rmsd > rmsd_delta:
                    raise ValueError("Molecules in SDF file could not be aligned.")

                pos = pose_lig.position
                roll, pitch, yaw = _rotation_matrix_to_euler(transform[:3, :3])

                v.append((roll, pitch, yaw, *pos, *pose_torsions))

        # RDLogger.EnableLog("rdApp.*")

        return lig, v

    def _fix_mol_valence(self, sanitize=True):
        Chem.SanitizeMol(
            self,
            sanitizeOps=(Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES),
        )

        for atom in self.GetAtoms():
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

        self.UpdatePropertyCache(strict=sanitize)

        if sanitize:
            Chem.SanitizeMol(self)
        return self

    # endregion

    # region Chemistry / topology analysis

    def __assign_atom_types(self):
        self._atom_types = np.array([AtomType.Unknown] * len(self.GetAtoms()))

        hba = [m[0] for m in self.GetSubstructMatches(HBA_STRUCT)]

        non_polar_h = [m[0] for m in self.GetSubstructMatches(NON_POLAR_H_STRUCT)]

        for i, atom in enumerate(self.GetAtoms()):
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

    def __get_center_atom(self):
        conf = self.GetConformer()
        centroid = Chem.rdMolTransforms.ComputeCentroid(conf)

        closest_dist = float("inf")
        closest_i = None
        for atom in self.GetAtoms():
            if atom.GetAtomicNum() > 1:
                i = atom.GetIdx()
                dist = np.linalg.norm(conf.GetAtomPosition(i) - centroid)

                if dist < closest_dist:
                    closest_dist = dist
                    closest_i = i

        return closest_i

    def __compute_rotatable_torsions(self, flex_hydrogens: bool = False):
        """
        Calculates the rotatable torsion angles and stores them along with their
        indices defining the torsion in the molecule. This includes identifying
        rotatable bonds, constructing torsion definitions, and determining torsion
        angles for each rotatable bond in the molecule. The results are stored as
        attributes for later use.

        """
        # num_rotatable_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(
        #     self, strict=True
        # )

        # self.__rotatable_torsions = np.empty(num_rotatable_bonds, dtype=object)
        # self.__torsion_angles = np.zeros(num_rotatable_bonds)
        rotatable_torsions = []
        torsion_angles = []

        rotatable_bonds = self.GetSubstructMatches(ROTATABLE_BOND_STRUCT)

        distance_matrix = np.array(Chem.GetDistanceMatrix(self))[self._center_atom, :]

        for _, b in enumerate(rotatable_bonds):
            i_atom_1 = b[0]
            i_atom_2 = b[1]

            if not flex_hydrogens:
                heavy_degree_atom_1 = sum(
                    [
                        1
                        for nbr in self.GetAtomWithIdx(i_atom_1).GetNeighbors()
                        if nbr.GetAtomicNum() > 1
                    ]
                )
                heavy_degree_atom_2 = sum(
                    [
                        1
                        for nbr in self.GetAtomWithIdx(i_atom_2).GetNeighbors()
                        if nbr.GetAtomicNum() > 1
                    ]
                )
                if heavy_degree_atom_1 == 1 or heavy_degree_atom_2 == 1:
                    continue

            atom_1_neighbors = self.GetAtomWithIdx(i_atom_1).GetNeighbors()
            atom_2_neighbors = self.GetAtomWithIdx(i_atom_2).GetNeighbors()

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
            # Dont care about:?
            torsion_angles.append(
                Chem.rdMolTransforms.GetDihedralRad(self.GetConformer(), *torsion)
            )

        self.__rotatable_torsions = rotatable_torsions
        self.__torsion_angles = torsion_angles

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
        return self.GetConformer().GetPositions()

    def get_positions(self, conf_id: int = -1) -> NDArray[np.float32]:
        """Returns the positions of all atoms in a specific conformer.

        Parameters
        ----------
        conf_id : int, default -1
             The conformer id to retrieve positions from. By default selects the global conformer.

        Returns
        -------
        list
        """
        return self.GetConformer(conf_id).GetPositions()

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
    def torsions(self) -> NDArray[np.float32]:
        """The torsion angles of the rotatable torsions in the molecule.

        Returns
        -------
        list
            A list containing all torsion angles in the molecule, in radians.

        """
        if not len(self.__rotatable_torsions) > 0:
            self.__compute_rotatable_torsions()
        for i, torsion in enumerate(self.__rotatable_torsions):
            self.__torsion_angles[i] = Chem.rdMolTransforms.GetDihedralRad(
                self.GetConformer(), *torsion
            )
        return self.__torsion_angles

    @property
    def n_tors(self):
        """The number of torsions in the molecule."""
        return self.layout.n_tors

    def set_torsion(self, i_torsion: int, angle_rad: float, conf_id: int = -1) -> None:
        """Set the torsion angle of a molecule :class:`~rdkit.Chem.rdchem.Conformer` for a
        specific torsion.

        Parameters
        ----------
        i_torsion : int
            The index of the rotatable torsion bond.
        angle_rad : float
            The torsion angle in radians.
        conf_id : int, default -1
            The conformer id to set the torsion to. By default selects the global conformer.
        """
        Chem.rdMolTransforms.SetDihedralRad(
            self.GetConformer(conf_id),
            *self.__rotatable_torsions[i_torsion],
            angle_rad,
        )

    def set_torsions(self, angles_rad: list[float], conf_id: int = -1) -> None:
        """Sets the torsion angles for a molecule.

        Parameters
        ----------
        angles_rad : array_like
            A list of float values representing torsion angles in radians.
        conf_id : int, default -1
            The conformer id to set the torsions to. By default selects the global conformer.
        """
        if len(angles_rad) > 0 and len(self.__rotatable_torsions) == 0:
            warnings.warn(
                "set_torsions called with angles but this Mol has no rotatable torsions. "
                "Did you forget flexible=True when constructing the Mol?",
                UserWarning,
                stacklevel=2,
            )
        for i, angle in enumerate(angles_rad):
            self.set_torsion(i, angle, conf_id)

    def transform(
        self,
        rotation: NDArray[np.float32],
        translation: NDArray[np.float32],
        conf_id: int = -1,
    ) -> None:
        """Transforms the conformer of the molecule with respect to the center atom's coordinates.

        Parameters
        ----------
        roll : float
            Roll angle of rotation in radians.
        pitch : float
            Pitch angle of rotation in radians.
        yaw :
            Yaw angle of rotation in radians.
        x : float
            Translation along the x-axis.
        y : float
            Translation along the y-axis.
        z : float
            Translation along the z-axis.

        conf_id : int
            The conformer id to transform. By default selects the global conformer.

        """
        conf = self.GetConformer(conf_id)

        if self.layout.rot_type == "euler":
            assert len(rotation) == 3, "Rotation must be a 3-vector for Ligand with euler rotation."
            rotate = _rotation_matrix_from_euler(*rotation)
        else:
            assert len(rotation) == 4, (
                "Rotation must be a 4-vector for Ligand with quaternion rotation."
            )
            rotate = _rotation_matrix_from_quat(*rotation)

        center_atom = conf.GetAtomPosition(self._center_atom)
        center_atom_coords = np.array(
            [center_atom.x, center_atom.y, center_atom.z],
        )

        transformation_matrix = _compose_delta_transform(
            rotate[:3, :3],
            translation,
            self.__cur_rotation_matrix[:3, :3],
            center_atom_coords,
        )

        if conf_id == -1:
            self.__cur_rotation_matrix = rotate[:3, :3]
            self.__cur_rotation = np.asarray(rotation)
        # else:
        #     transformation_matrix = new_transform
        #     transformation_matrix[:3, 3] -= new_transform[:3, :3] @ center_atom_coords

        Chem.rdMolTransforms.TransformConformer(conf, transformation_matrix)

    def update(self, pose: Pose | NDArray, new_conf: bool = False) -> int:
        """Update the molecule with the new variables.

        Input should be shaped like ``(6 + n_tors,)``.

        This method can act either on the default :class:`~rdkit.Chem.rdchem.Conformer`,
        or can create a new conformer, apply the update and return the new conformers id.

        .. note::
            Using this method to create a new conformer on update is recommended. This allows for
            parallelization, as each thread is able to use their own conformer.
            See: TODO

        Parameters
        ----------
        new_vars : array_like
            Array containing the new variables in the order of
            ``(roll, pitch, yaw, x, y, z, *torsions)``.

        new_conf : bool, default False
            Whether to create a new conformer to apply the update to.

        Returns
        -------
        int
            Conformer id of the updated molecule. If no new conformer is created, returns -1,
            which is the id of the global conformer.

        """
        if isinstance(pose, Pose):
            assert pose.layout == self.layout, "Pose and molecule layout do not match."
        else:
            assert len(pose) == self.layout.rot_dim + 3 + self.layout.n_tors, (
                "Pose parameterization does not match molecule layout."
            )
            pose = Pose(pose, self.layout)

        conf_id = -1
        if new_conf:
            conf_id = self.AddConformer(self.GetConformer(), assignId=True)

        self.transform(pose.rotation, pose.translation, conf_id=conf_id)
        self.set_torsions(pose.torsions, conf_id=conf_id)

        return conf_id

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
        self.transform(self.layout.identity_rotation, np.zeros(3))
        self._center_atom = value
        self.transform(self.layout.identity_rotation, np.zeros(3))

    @property
    def position(self):
        """The position of the molecule.

        This is equal to the position of the center atom.

        Returns
        -------
        list
            A list of shape (3,) containing the x, y, and z coordinates of the center atom.
        """
        center_atom_coords = self.GetConformer().GetAtomPosition(self._center_atom)
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

        new_mol = Chem.Mol(self)

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

    def to_sdf(self, file: str | SDWriter, conf_id: int = -1):
        """Write the current molecule to an SDF file.

        Parameters
        ----------
        file : str, ~rdkit.Chem.rdmolfiles.SDWriter
            Either a filename of a file to create, or an open
            :class:`~rdkit.Chem.rdmolfiles.SDWriter` object to write to.
        conf_id : int, optional
            The conformer id to write.

        """
        close_writer = False
        if isinstance(file, str):
            writer = SDWriter(file)
            close_writer = True
        elif isinstance(file, SDWriter):
            writer = file
        else:
            raise ValueError("file must be either filename str or SDWriter")

        writer.write(self, confId=conf_id)

        if close_writer:
            writer.close()

    def v_to_sdf(self, file: str, poses: Poses):
        """Write the current molecule with positions `v` to an SDF file.

        Parameters
        ----------
        file : str
            The path to the file to create.
        v : array_like
            An array of shape ``(6 + n_tors, n)``, containing molecular positions to write.

        """
        writer = Chem.SDWriter(file)

        for pose in poses:
            conf_id = self.update(pose, new_conf=True)
            self.to_sdf(writer, conf_id=conf_id)
            self.RemoveConformer(conf_id)
        writer.close()

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
            else self.GetNumHeavyAtoms() >= VIEWER_PROTEINS_HEAVY_ATOMS_CUTOFF
        )

        if is_protein:
            m_id = self._viewer_add_prot(viewer, c_m_id, l_options)
        else:
            mblock = Chem.MolToMolBlock(self)
            viewer.view.addModel(mblock, "mol")
            m_id = c_m_id + 1
            viewer.view.setStyle(
                {"model": m_id}, {"stick": {"colorscheme": l_options["colorscheme"]}}
            )

            viewer._set_hover(self, m_id, self.draw_options)  # noqa
        return m_id

    def _viewer_add_prot(self, viewer, c_m_id, options: dict = None):
        pdbblock = Chem.MolToPDBBlock(self)

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
                for a in self.GetAtoms():
                    a.SetProp("atomNote", f"{a.GetIdx()}")
            case "type":
                for a in self.GetAtoms():
                    ty = self._atom_types[a.GetIdx()]
                    a.SetProp("atomNote", f"{str(ty)}")
            case _:
                for a in self.GetAtoms():
                    a.ClearProp("atomNote")

        highlight = []
        if "highlight" in self.draw_options:
            match self.draw_options["highlight"].lower():
                case "center":
                    highlight = [self._center_atom]
                case list():
                    highlight = self.draw_options["highlight"]

        d2d.DrawMolecule(self, highlightAtoms=highlight)
        d2d.FinishDrawing()
        return d2d.GetDrawingText()

    # endregion

    # region Dunder / misc

    def __hash__(self):
        # TODO!
        return id(self)

    # endregion


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

    def __array__(self, dtype=None):
        return self._v if dtype is None else self._v.astype(dtype)

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

    def __array__(self, dtype=None):
        return self._vs if dtype is None else self._vs.astype(dtype)

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
    def from_poses(cls, poses: list[Pose], layout: PoseLayout | None = None):
        return cls(np.stack([np.asarray(p) for p in poses]), layout or poses[0].layout)
