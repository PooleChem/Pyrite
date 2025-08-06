from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING

import numpy as np
import py3Dmol
from IPython.display import SVG, Image
from numpy.typing import NDArray
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, SDWriter, RegistrationHash

from .view import Viewer
from .atom_consts import AtomType, vina_atom_consts

if TYPE_CHECKING:
    from pyrite.bounds import Bounds


def _rotation_matrix_to_euler(r: NDArray):
    """
    Extract ZYX (yaw-pitch-roll) Euler angles from a 3×3 rotation matrix R.
    Returns (phi, theta, psi) = (roll, pitch, yaw) in radians.
    """
    # clamp to handle numerical errors outside [-1,1]
    sy = -r[2, 0]
    theta = np.arcsin(np.clip(sy, -1.0, 1.0))

    # Check for gimbal lock
    # if np.isclose(np.cos(theta), 0.0):
    #     # Gimbal lock: pitch is ±90°
    #     # Roll and yaw are coupled; set roll=0 and compute yaw:
    #     phi = 0.0
    #     psi = np.arctan2(-R[0, 1], R[1, 1])
    # else:
    phi = np.arctan2(r[2, 1], r[2, 2])  # roll
    psi = np.arctan2(r[1, 0], r[0, 0])  # yaw

    return phi, theta, psi


# TODO: Support Quaternions
def _rotation_matrix(roll: float, pitch: float, yaw: float) -> NDArray:
    r"""Create a rotation matrix from Euler angles.

    Create a 4x4 rotation matrix from Euler angles (roll, pitch, yaw) in radians.


    Parameters
    ----------
    roll : float
        Roll angle in radians.

    pitch : float
        Pitch angle in radians.

    yaw : float
        Yaw angle in radians.

    Returns
    -------
    rotation_matrix : NDArray
        A rotation matrix of shape (4, 4) representing the rotation transformation for the specified Euler angles.
    """

    sin_roll = np.sin(roll)
    cos_roll = np.cos(roll)
    sin_pitch = np.sin(pitch)
    cos_pitch = np.cos(pitch)
    sin_yaw = np.sin(yaw)
    cos_yaw = np.cos(yaw)

    rotation_matrix = np.array(
        [
            [
                cos_yaw * cos_pitch,
                cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll,
                cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll,
            ],
            [
                sin_yaw * cos_pitch,
                sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll,
                sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll,
            ],
            [
                -sin_pitch,
                cos_pitch * sin_roll,
                cos_pitch * cos_roll,
            ],
        ]
    )
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    return transformation_matrix


def _translation_matrix(x: float, y: float, z: float) -> NDArray:
    """Create a translation matrix from coordinates.

    Create a 4x4 translation matrix from the provided x, y, and z coordinates.

    Parameters
    ----------
    x, y, z : float
        Coordinates for the translation.

    Returns
    -------
    translation_matrix : NDArray
    """
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = np.array([x, y, z])
    return translation_matrix


def _fix_valence(mol, sanitize=True):
    Chem.SanitizeMol(
        mol,
        sanitizeOps=(
            Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
        ),
    )

    for atom in mol.GetAtoms():
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

    mol.UpdatePropertyCache(strict=sanitize)

    if sanitize:
        Chem.SanitizeMol(mol)
    return mol


class Ligand(Chem.Mol):
    """
    Representation of a ligand.

    The Ligand class provides methods for initializing molecules from various sources,
    such as SMILES strings, PDB files, or objects. It also assigns atom
    types, rotatable dihedrals, and the center point of the molecule. Furthermore, it allows for
    easy manipulation of ligand position, rotation, and torsion angles.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The :class:`~rdkit.Chem.rdchem.Mol` object representing the molecule.
    center_atom : int, optional
        The index of the atom to use as the center point for rotations.
        By default, the center atom is determined automatically by selecting the heavy atom closest
        to the initial conformer centroid.


    """

    __default_draw_options = {
        "size": (400, 300),
        "colorPalette": "default",
        "note": "",
        "highlight": "",
        "colorscheme": "default",
    }

    def __new__(cls, mol: Chem.Mol = None, **kwargs):
        if mol is None:
            inst = super().__new__(cls)
        else:
            inst = Chem.Mol(mol)
            inst.__class__ = cls

        if inst.GetNumConformers() == 0:
            params = Chem.AllChem.ETKDGv3()
            params.randomSeed = 0xC0FFEE
            Chem.AllChem.EmbedMolecule(inst, params)
        return inst

    def __init__(
        self,
        mol: Chem.Mol = None,
        center_atom: int = None,
        flex_hydrogens: bool = False,
        rigid: bool = False,
    ):
        # NO SUPER.__INIT__, that breaks stuff
        self.__cur_transform = np.eye(4)
        self.__cur_rotation = [0, 0, 0]
        self.__rotatable_dihedrals = np.array([], dtype=object)
        self.__dihedral_angles = np.array([])

        self._center_atom = (
            center_atom if center_atom is not None else self.__get_center_atom()
        )

        if not rigid:
            self.__compute_rotatable_dihedrals(flex_hydrogens)
        self.__assign_atom_types()
        Chem.rdPartialCharges.ComputeGasteigerCharges(self)

        self._init_state = (
            copy.deepcopy(self.__cur_rotation),
            copy.deepcopy(self.position),
            copy.deepcopy(self.__dihedral_angles),
        )

        self.draw_options = self.__default_draw_options.copy()

    @classmethod
    def from_smiles(cls, smiles: str, **kwargs):
        """Constructs an instance of :class:`Ligand` from a SMILES string representation of a molecule.

        Parameters
        ----------
        smiles : str
            The SMILES string representation of the molecule.

        Returns
        -------
        Ligand
        """
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        mol = _fix_valence(mol)
        mol = Chem.AllChem.AddHs(mol, addCoords=True)
        return cls(mol, **kwargs)

    @classmethod
    def from_rdkit(cls, mol: Chem.Mol, **kwargs):
        """Create an instance of :class:`Ligand` from an RDKit :class:`~rdkit.Chem.rdchem.Mol` object.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            The input molecule as an RDKit :class:`~rdkit.Chem.rdchem.Mol` object.

        Returns
        -------
        Ligand
        """
        return cls(mol, **kwargs)

    @classmethod
    def from_pdb(
        cls,
        pdb_file: str,
        template_smiles: str = None,
        template_sdf: str = None,
        **kwargs,
    ):
        r"""Creates an instance of :class:`Ligand` from a PDB file.

        Read a PDB file to generate a molecule object, add hydrogens
        (including their coordinates), and assign stereochemical information based on the
        3D structure of the molecule and the supplied template. The template can be either
        a SMILES string or an SDF file, and is needed to ensure the correct bond order assignment.

        Parameters
        ----------
        pdb_file : str
            Path to the PDB file containing the molecule.
        template_smiles : str, optional
            SMILES string representing a reference molecule. Either `template_smiles` or
            `template_sdf` must be provided.
        template_sdf : str, optional
            Path to an SDF file containing a reference molecule. Either `template_smiles` or
            `template_sdf` must be provided.

        Returns
        -------
        Ligand
        """
        if template_smiles is None and template_sdf is None:
            raise ValueError("Template required for PDB input.")
        if template_sdf is not None and template_smiles is not None:
            raise ValueError("Only one template allowed.")
        mol = Chem.MolFromPDBFile(pdb_file, sanitize=False, removeHs=False)

        if template_smiles is not None:
            template = Chem.MolFromSmiles(template_smiles)
        else:
            template = Chem.MolFromMolFile(template_sdf)
        mol = Chem.AllChem.AssignBondOrdersFromTemplate(template, mol)

        n_a_orig = mol.GetNumAtoms()
        mol = Chem.AllChem.AddHs(mol, addCoords=True)
        n_a_new = mol.GetNumAtoms()

        if n_a_orig != n_a_new:
            warnings.warn(
                f"Added {n_a_new - n_a_orig} hydrogens to molecule from PDB file.",
            )

        Chem.AssignStereochemistryFrom3D(mol)
        Chem.SanitizeMol(mol)

        return cls(mol, **kwargs)

    # TODO: be able to load and return multiple ligands from the same SDF file. (for v_from_sdf too)
    @classmethod
    def from_sdf(cls, mol_file: str, **kwargs):
        """Creates an instance of :class:`Ligand` from an SDF file.

        Parameters
        ----------
        mol_file : str
            Path to the SDF file containing the molecule.

        Returns
        -------
        Ligand
        """
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromMolFile(
            mol_file, removeHs=False, strictParsing=False, sanitize=False
        )
        RDLogger.EnableLog("rdApp.*")
        mol = _fix_valence(mol)
        mol = Chem.AllChem.AddHs(mol, addCoords=True)
        return cls(mol, **kwargs)

    @classmethod
    def v_from_sdf(cls, mol_file: str, rmsd_delta: float = 0.5, **kwargs):
        """Creates a :class:`Ligand` from an SDF file, and also returns a list of variables
        representing the conformations in the SDF file.

        Parameters
        ----------
        mol_file : str
            Path to the SDF file containing the molecule, with multiple conformations.
            All molecules in the SDF file should be equal.
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
        Ligand
        variables : numpy.ndarray
            The conformations in the SDF file, represented by a tuple of size ``(6 + n_dihedrals)``,
            as expected by e.g. :meth:`update`.
        """
        # RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromMolFile(
            mol_file, removeHs=False, strictParsing=False, sanitize=False
        )
        mol = _fix_valence(mol)
        mol = Chem.AllChem.AddHs(mol, addCoords=True)
        canon_smiles = Chem.CanonSmiles(Chem.MolToSmiles(mol))
        lig = cls(mol, **kwargs)

        # Alignment atom map
        matches = lig.GetSubstructMatches(lig, uniquify=True, useChirality=True)
        atom_map = [list(zip(range(lig.GetNumAtoms()), match)) for match in matches]

        atom_map = [t for sub in atom_map for t in sub]  # Flatten

        v = []
        with Chem.SDMolSupplier(mol_file, removeHs=False, sanitize=False) as supl:
            for pose in supl:
                pose = _fix_valence(pose)
                pose = Chem.AllChem.AddHs(pose, addCoords=True)
                if Chem.CanonSmiles(Chem.MolToSmiles(pose)) != canon_smiles:
                    raise ValueError("Molecules in SDF file are not equal.")
                # TODO: dont create whole ligand every time.
                pose_lig = cls(pose, **kwargs)
                pose_dihedrals = pose_lig.dihedral_angles

                # Set dihedrals equal for alignment
                pose_lig.set_dihedral_angles(lig.dihedral_angles)

                # Align molecule
                rmsd, transform = Chem.rdMolAlign.GetAlignmentTransform(
                    lig, pose_lig, atomMap=atom_map
                )
                if rmsd > rmsd_delta:
                    raise ValueError("Molecules in SDF file could not be aligned.")

                pos = pose_lig.position
                roll, pitch, yaw = _rotation_matrix_to_euler(transform[:3, :3])

                v.append((roll, pitch, yaw, *pos, *pose_dihedrals))

        # RDLogger.EnableLog("rdApp.*")

        return lig, v

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

        mblock = Chem.MolToMolBlock(self)
        viewer.view.addModel(mblock, "mol")
        m_id = c_m_id + 1
        viewer.view.setStyle(
            {"model": m_id}, {"stick": {"colorscheme": l_options["colorscheme"]}}
        )

        # if "note" not in self.draw_options:
        #     self.draw_options["note"] = ""
        # match self.draw_options["note"].lower():
        #     case "idx":
        #         hover_js_callback = Viewer._HOVER_LABEL_IDX_JS_CALLBACK
        #     case "type":
        #         hover_js_callback = (
        #             """function(atom,viewer,event,container) {
        #             let atom_types = """
        #             + str([AtomType(t).name for t in self.atom_types])
        #             + """
        #             if(!atom.label) {
        #                 atom.label = viewer.addLabel(atom_types[atom.index],{position: atom, backgroundColor: 'mintcream', fontColor:'black'});
        #             }}"""
        #         )
        #     case _:
        #         hover_js_callback = None
        #
        # if hover_js_callback is not None:
        #     view.setHoverable(
        #         {"model": m_id},
        #         True,
        #         hover_js_callback,
        #         Viewer._UNHOVER_LABEL_JS_CALLBACK,
        #     )
        viewer._set_hover(self, m_id, self.draw_options)  # noqa

        return m_id

    def _repr_png_(self):
        return self.__repr_picture(Draw.MolDraw2DCairo(*self.draw_options["size"]))

    def _repr_svg_(self):
        return self.__repr_picture(Draw.MolDraw2DSVG(*self.draw_options["size"]))

    def _repr_html_(self):
        return Viewer(  # noqa
            self,
            width=self.draw_options["size"][0],
            height=self.draw_options["size"][1],
        )._repr_html_()

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

    def __assign_atom_types(self):
        self._atom_types = np.array([AtomType.Unknown] * len(self.GetAtoms()))

        hba_struct = Chem.MolFromSmarts(
            "[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0])]"
        )
        hba = [m[0] for m in self.GetSubstructMatches(hba_struct)]

        non_polar_h_struct = Chem.MolFromSmarts("[#1;$([#1]-[#6,#14])]")
        non_polar_h = [m[0] for m in self.GetSubstructMatches(non_polar_h_struct)]

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

    def __compute_rotatable_dihedrals(self, flex_hydrogens: bool = False):
        """
        Calculates the rotatable dihedral angles and stores them along with their
        indices defining the dihedral in the molecule. This includes identifying
        rotatable bonds, constructing dihedral definitions, and determining dihedral
        angles for each rotatable bond in the molecule. The results are stored as
        attributes for later use.

        """

        # num_rotatable_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(
        #     self, strict=True
        # )

        # self.__rotatable_dihedrals = np.empty(num_rotatable_bonds, dtype=object)
        # self.__dihedral_angles = np.zeros(num_rotatable_bonds)
        rotatable_dihedrals = []
        dihedral_angles = []

        rotatable_bond_struct = Chem.MolFromSmarts(
            "[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])"
            "([CH3])[CH3])&!$([CD3](=[N,O,S])-!@[#7,O,S!D1])&!$([#7,O,S!D1]-!@[CD3]"
            "=[N,O,S])&!$([CD3](=[N+])-!@[#7!D1])&!$([#7!D1]-!@[CD3]=[N+])]-,:;!@"
            "[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])"
            "([CH3])[CH3])]"
        )

        rotatable_bonds = self.GetSubstructMatches(rotatable_bond_struct)

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

            ix_atom_1_neighbors = [
                a.GetIdx() for a in atom_1_neighbors if a.GetIdx() != i_atom_2
            ]
            ix_atom_2_neighbors = [
                a.GetIdx() for a in atom_2_neighbors if a.GetIdx() != i_atom_1
            ]

            dihedral = (
                min(ix_atom_1_neighbors),
                i_atom_1,
                i_atom_2,
                min(ix_atom_2_neighbors),
            )

            # (a, b, c, d)  |    o (center atom)
            # Als center_atom dichter bij b -> draait niet.
            # Als center_atom dichter bij c -> draait wel -> invert dihedral.
            # print(distance_matrix[dihedral[1]], distance_matrix[dihedral[2]])
            if distance_matrix[dihedral[2]] < distance_matrix[dihedral[1]]:
                dihedral = dihedral[::-1]

            rotatable_dihedrals.append(dihedral)
            # Dont care about:?
            dihedral_angles.append(
                (Chem.rdMolTransforms.GetDihedralRad(self.GetConformer(), *dihedral))
            )

        self.__rotatable_dihedrals = rotatable_dihedrals
        self.__dihedral_angles = dihedral_angles

    @property
    def atom_types(self):
        """The atom types of all atoms in the ligand.

        Returns
        -------
        numpy.ndarray
        """
        return self._atom_types

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
    def rotatable_dihedrals(self):
        """The rotatable dihedrals of the molecule.

        Returns
        -------
        list
            A list containing all rotatable dihedrals in the molecule,
            indicated by four atom indices. An entry looks like
            ``[i, j, k, l]``, where the rotated bond is between
            ``j`` and ``k``, and all atoms attached to ``k`` are moved.

        """
        return self.__rotatable_dihedrals

    @property
    def dihedral_angles(self) -> NDArray[np.float32]:
        """The dihedral angles of the rotatable dihedrals in the molecule.

        Returns
        -------
        list
            A list containing all dihedral angles in the molecule, in radians.

        """
        if not len(self.__rotatable_dihedrals) > 0:
            self.__compute_rotatable_dihedrals()
        for i, dihedral in enumerate(self.__rotatable_dihedrals):
            self.__dihedral_angles[i] = Chem.rdMolTransforms.GetDihedralRad(
                self.GetConformer(), *dihedral
            )
        return self.__dihedral_angles

    def set_dihedral_angle(
        self, i_dihedral: int, angle_rad: float, conf_id: int = -1
    ) -> None:
        """Set the dihedral angle of a molecule :class:`~rdkit.Chem.rdchem.Conformer` for a
        specific dihedral.

        Parameters
        ----------
        i_dihedral : int
            The index of the rotatable dihedral bond.
        angle_rad : float
            The dihedral angle in radians.
        conf_id : int, default -1
            The conformer id to set the dihedral to. By default selects the global conformer.
        """
        Chem.rdMolTransforms.SetDihedralRad(
            self.GetConformer(conf_id),
            *self.__rotatable_dihedrals[i_dihedral],
            angle_rad,
        )

    def set_dihedral_angles(self, angles_rad: list[float], conf_id: int = -1) -> None:
        """Sets the dihedral angles for a molecule.

        Parameters
        ----------
        angles_rad : array_like
            A list of float values representing dihedral angles in radians.
        conf_id : int, default -1
            The conformer id to set the dihedrals to. By default selects the global conformer.
        """
        for i, angle in enumerate(angles_rad):
            self.set_dihedral_angle(i, angle, conf_id)

    def transform(
        self,
        roll: float,
        pitch: float,
        yaw: float,
        x: float,
        y: float,
        z: float,
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

        rotate = _rotation_matrix(roll, pitch, yaw)
        translate = _translation_matrix(x, y, z)

        new_transform = translate @ rotate

        center_atom = conf.GetAtomPosition(self._center_atom)
        center_atom_coords = np.array(
            [center_atom.x, center_atom.y, center_atom.z],
        )

        self.__cur_transform[:3, 3] = center_atom_coords

        reverse = np.linalg.inv(self.__cur_transform)

        transformation_matrix = new_transform @ reverse

        if conf_id == -1:
            self.__cur_transform = new_transform
            self.__cur_rotation = [roll, pitch, yaw]
        # else:
        #     transformation_matrix = new_transform
        #     transformation_matrix[:3, 3] -= new_transform[:3, :3] @ center_atom_coords

        Chem.rdMolTransforms.TransformConformer(conf, transformation_matrix)

    def update(self, new_vars: NDArray, new_conf: bool = False) -> int:
        """Update the molecule with the new variables.

        Input should be shaped like ``(6 + n_dihedrals,)``.

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
            ``(roll, pitch, yaw, x, y, z, *dihedrals)``.

        new_conf : bool, default False
            Whether to create a new conformer to apply the update to.

        Returns
        -------
        int
            Conformer id of the updated molecule. If no new conformer is created, returns -1,
            which is the id of the global conformer.

        """
        assert len(new_vars) == 6 + len(self.__rotatable_dihedrals)

        conf_id = -1
        if new_conf:
            conf_id = self.AddConformer(self.GetConformer(), assignId=True)

        transformation = new_vars[:6]
        dihedrals = new_vars[6:]

        self.transform(*transformation, conf_id=conf_id)
        self.set_dihedral_angles(dihedrals, conf_id=conf_id)

        return conf_id

    # TODO: breaks if center atom is changed after init.
    def reset(self):
        """Resets the ligand to its initial state.

        Resets ligand rotation, position, and dihedral angles to the state upon initialization.
        """
        self.transform(*self._init_state[0], *self._init_state[1])
        self.set_dihedral_angles(self._init_state[2])

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
        self.transform(0, 0, 0, 0, 0, 0)
        self._center_atom = value
        self.transform(0, 0, 0, 0, 0, 0)

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

    def place_in(
        self,
        binding_site: Bounds,
        n_positions: int,
        n_conformations: int,
        placement: str = "random",
        conformations: str = "conformer",
        combine: str = "random",
    ) -> NDArray:
        """Retrieve a list of random placements and conformers for the molecule in the binding_site.


        Parameters
        ----------
        binding_site : Bounds
            The binding site in which to place the ligand.
        n_positions : int
            The number of ligand positions to generate. When `placement` = ``grid``, this is the
            size of the grid in every axis. For example, an `n_positions` of 4 would yield
            :math:`4^3` positions.
        n_conformations : int
            The number of conformations to generate.
        placement : {'random', 'grid'}, default 'random'
            The placement method to use. ``random`` places the ligand randomly in the binding site.
            ``grid`` creates a grid in the binding site.
        conformations : {'conformer', 'random}, default 'conformer'
            The conformer generation method to use. ``conformer`` will create conformers using
            RDKit :func:`~rdkit.Chem.rdDistGeom.EmbedMultipleConfs`. ``random`` will set
            all dihedral angles to random values. This is faster, but can create
            physically impossible configurations.
        combine : {'random', 'grid'}, default 'random'
            The combination method. ``random`` will create random combinations of positions and
            dihedral angles. When ``n_positions >= n_conformations``, a random conformation is
            chosen for every position, and the other way around. This results in an output size of
            ``max(n_positions, n_conformations)``.
            ``grid`` combines all positions with all conformations, resulting in an output size of
            ``n_positions * n_conformations``.


        Returns
        -------
        numpy.ndarray
            An array of shape ``(max(n_positions, n_conformations), 6 + n_dihedrals)`` when
            `combine` is ``random``, or shape
            ``(n_positions * n_conformations, 6 + n_dihedrals)`` when `combine` is ``grid``.
        """

        if placement not in {"random", "grid"}:
            raise ValueError("placement must be either 'random' or 'grid'")
        if conformations not in {"conformer", "random"}:
            raise ValueError("conformations must be either 'conformer' or 'random'")
        if combine not in {"random", "grid"}:
            raise ValueError("combine must be either 'random' or 'grid'")

        positions = []
        if placement == "random":
            positions = binding_site.place_random_uniform(n_positions)
        elif placement == "grid":
            positions = binding_site.place_grid(n_positions)

        dihedrals = []
        if conformations == "conformer":
            dihedrals = self.get_n_conformer_dihedral_configurations(n_conformations)
        elif conformations == "random":
            dihedrals = self.get_n_random_dihedral_configurations(n_conformations)

        out_pos, out_dih = [], []
        if combine == "random":
            if positions.shape[0] >= dihedrals.shape[0]:
                sel_dihedrals = np.random.choice(
                    dihedrals.shape[0], size=positions.shape[0], replace=True
                )
                out_pos = positions
                out_dih = dihedrals[sel_dihedrals]
            else:
                sel_positions = np.random.choice(
                    positions.shape[0], size=dihedrals.shape[0], replace=True
                )
                out_pos = positions[sel_positions]
                out_dih = dihedrals

        elif combine == "grid":
            out_pos = np.repeat(positions, dihedrals.shape[0], axis=0)
            out_dih = np.tile(dihedrals, (positions.shape[0], 1))

        return np.concatenate((out_pos, out_dih), axis=1)

    # TODO: not too keen on this placement
    def get_n_random_dihedral_configurations(self, n: int) -> NDArray:
        """Retrieve a list of `n` random dihedral configurations for the molecule.

        .. warning::
            This returns lists of truly random dihedral configurations, and may thus result
            in physically impossible configurations.

        Parameters
        ----------
        n : int
            The number of configurations to generate.


        Returns
        -------
        numpy.ndarray
            An array of shape ``(n_dihedrals, n)`` containing `n` dihedral configurations.
        """
        return np.random.rand(n, len(self.__rotatable_dihedrals)) * 2 * np.pi - np.pi

    def get_n_conformer_dihedral_configurations(self, n: int) -> NDArray:
        """Retrieve a list of `n` conformation-based dihedral configurations for the molecule.

        .. note::
            This method does not necessarily result in unique configurations.


        Parameters
        ----------
        n : int
            The number of configurations to generate.


        Returns
        -------
        numpy.ndarray
            An array of shape ``(n_dihedrals, n)`` containing `n` dihedral configurations.
        """
        params = Chem.AllChem.ETKDGv3()
        params.randomSeed = 0xC0FFEE

        new_mol = Chem.Mol(self)

        cids = Chem.AllChem.EmbedMultipleConfs(new_mol, n, params)

        configurations = np.empty((n, len(self.__rotatable_dihedrals)))
        for i, cid in enumerate(cids):
            dihedral_angles = np.zeros(len(self.__rotatable_dihedrals))
            for j, dihedral in enumerate(self.__rotatable_dihedrals):
                dihedral_angles[j] = Chem.rdMolTransforms.GetDihedralRad(
                    new_mol.GetConformer(cid), *dihedral
                )
            configurations[i] = dihedral_angles

        return configurations

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

    def v_to_sdf(self, file: str, v: NDArray):
        """Write the current molecule with positions `v` to a file.

        Parameters
        ----------
        file : str
            The path to the file to create.
        v : array_like
            An array of shape ``(6 + n_dihedrals, n)``, containing molecular positions to write.

        """

        writer = Chem.SDWriter(file)
        # print(v)
        for var in v:
            conf_id = self.update(var, new_conf=True)
            self.to_sdf(writer, conf_id=conf_id)
            self.RemoveConformer(conf_id)
        writer.close()


class Receptor:
    """
    Representation of a receptor.

    The Receptor class provides methods for initializing receptors from various sources *(not yet)*
    and includes functionality for assigning atom types.

    This class currently holds an :class:`rdkit.Chem.rdchem.Mol` object, which is used
    to obtain positions and atom types from the pdb file.

    .. note::
        When loading a PDB file, make sure to have prepared the file beforehand, by removing water,
        ensuring that there are no missing heavy atoms or residues, and adding hydrogen.


    Parameters
    ----------
    pdb_path : str
        The path of the pdb file containing the receptor structure.

    """

    __default_draw_options = {
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
    # __default_surface_options = {
    #     "type": "MS",
    #     "color": "lightgray",
    #     "opacity": 0.9,
    # }

    def __init__(self, mol, path: str):
        self._path = path
        self._rdkit = mol

        # self._openff = Topology.from_pdb(pdb_path)
        self.__assign_atom_types_rdkit()
        Chem.rdPartialCharges.ComputeGasteigerCharges(self._rdkit)

        #
        # self._positions = np.array(self.openff.get_positions().m_as(unit.angstrom))
        self._positions = np.array(self._rdkit.GetConformer().GetPositions())

        mol_layers = RegistrationHash.GetMolLayers(self._rdkit)
        self._prehash = hash(RegistrationHash.GetMolHash(mol_layers))

        self.draw_options = self.__default_draw_options.copy()
        # self.surface_options = self.__default_surface_options.copy()

    @classmethod
    def from_pdb(cls, pdb_path: str):
        """Constructs a :class:`Receptor` from a pdb file.

        .. note::
            When loading a PDB file, make sure to have prepared the file beforehand, by removing water,
            ensuring that there are no missing heavy atoms or residues, and adding hydrogen.


        Parameters
        ----------
        pdb_path : str
            The path of the pdb file containing the receptor structure.


        Returns
        -------
        Receptor

        """
        mol = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=False)
        rdkit = Chem.AllChem.AddHs(_fix_valence(mol, sanitize=False))
        return cls(rdkit, pdb_path)

    @classmethod
    def from_sdf(cls, sdf_path: str):
        """Load a receptor from an SDF file.

        Parameters
        ----------
        sdf_path : str
            The path of the SDF file containing the receptor structure.

        Returns
        -------
        Receptor

        """
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromMolFile(
            sdf_path, removeHs=False, strictParsing=False, sanitize=False
        )
        RDLogger.EnableLog("rdApp.*")
        mol = _fix_valence(mol, sanitize=False)
        mol = Chem.AllChem.AddHs(mol, addCoords=True)
        return cls(mol, sdf_path)

    def __assign_atom_types_rdkit(self):
        self._atom_types = np.array([AtomType.Unknown] * len(self._rdkit.GetAtoms()))

        hba_struct = Chem.MolFromSmarts(
            "[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0])]"
        )
        hba = [m[0] for m in self._rdkit.GetSubstructMatches(hba_struct)]

        non_polar_h_struct = Chem.MolFromSmarts("[#1;$([#1]-[#6,#14])]")
        non_polar_h = [
            m[0] for m in self._rdkit.GetSubstructMatches(non_polar_h_struct)
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

    def __assign_atom_types(self):
        self._atom_types = np.array([AtomType.Unknown] * self.openff.n_atoms)

        # THIS IS SOMETIMES TIMING OUT:
        hba = self.openff.chemical_environment_matches(
            "[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0:1]):1]",
            unique=True,
        )
        hba = [m.topology_atom_indices[0] for m in hba]

        non_polar_h = self.openff.chemical_environment_matches(
            "[#1;$([#1]-[#6,#14]):1]",
            unique=True,
        )
        non_polar_h = [m.topology_atom_indices[0] for m in non_polar_h]

        for i, atom in enumerate(self.openff.atoms):
            atomic_number = atom.atomic_number

            a_str = atom.symbol
            if atomic_number == 1 and self.openff.atom_index(atom) not in non_polar_h:
                a_str = "HD"
            elif atomic_number == 6 and atom.is_aromatic:
                a_str = "A"
            elif atomic_number == 8:
                a_str = "OA"
            elif atomic_number == 7 and self.openff.atom_index(atom) in hba:
                a_str = "NA"
            elif atomic_number == 16 and self.openff.atom_index(atom) in hba:
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
            for neigh in atom.bonded_atoms:
                if neigh.symbol == "H":
                    hbonded = True
                elif neigh.symbol != "C":
                    heterobonded = True

            a_type = a_type.adjust(hbonded, heterobonded)

            self._atom_types[i] = a_type

    def __assign_residue_ids(self):
        self.residues = [""] * len(self._rdkit.GetAtoms())
        for i, atom in enumerate(self._rdkit.GetAtoms()):
            res_info = str(atom.GetPDBResidueInfo().GetResidueName()) + str(
                atom.GetPDBResidueInfo().GetResidueId()
            )
            self.residues[i] = res_info

    # @property
    # def openff(self):
    #     """The openff :class:`~openff.toolkit.topology.Topology` object.
    #
    #     Returns
    #     -------
    #     openff.toolkit.topology.Topology
    #     """
    #     return self._openff

    @property
    def rdkit(self):
        """The rdkit :class:`~rdkit.Chem.rdchem.Mol` object.

        Returns
        -------
        rdkit.Chem.rdchem.Mol
        """
        return self._rdkit

    @property
    def positions(self):
        """The positions of all atoms in the receptor.

        Returns
        -------
        numpy.ndarray
        """
        return self._positions

    def get_positions(self):
        """Get the positions of all atoms in the receptor.

        Returns
        -------
        numpy.ndarray
        """
        return self._positions

    @property
    def atom_types(self):
        """The atom types of all atoms in the receptor.

        Returns
        -------
        numpy.ndarray
        """
        return self._atom_types

    def atom_type(self, atom_index: int, mask: NDArray = None):
        """Get the atom type of the specified atom, optionally taking `mask` into account.

        Parameters
        ----------
        atom_index : int
            The atom id for which to get the atom type.
        mask : numpy.ndarray, optional
            An array by which to mask the receptor atoms before indexing. Useful in combination
            with e.g. :class:`~pyrite.scoring.dependencies.KNNDependency` on a masked receptor.

        Returns
        -------
        AtomType
        """
        if mask is None:
            mask = np.full_like(self._atom_types, True)
        return (self._atom_types[mask])[atom_index]

    def __hash__(self):
        # return hash(self._path)
        return self._prehash

    def set_draw_options(
        self,
        options,
    ):
        """Set draw options.

        Parameters
        ----------
        options : dictionary
            The options to apply.

        """

        # unknown = set(options) - set(self.draw_options)
        # unknown.update(set(surface) - set(self.surface_options))
        # if unknown:
        #     raise ValueError(f"Unknown options: {unknown}")

        self.draw_options.update(options)
        # self.surface_options.update(surface)

    def _viewer_add_(self, viewer, c_m_id, options=None):
        if options is None:
            options = {}
        l_options = self.draw_options.copy()
        l_options.update(options)

        pdbblock = Chem.MolToPDBBlock(self._rdkit)

        viewer.view.addModel(pdbblock, "pdb")
        m_id = c_m_id + 1
        viewer.view.setStyle(
            {"model": m_id},
            {
                "cartoon": {
                    "color": l_options["color"],
                    "style": l_options["style"],
                    "hidden": l_options["hideprotein"],
                },
            },
        )

        for res in l_options["stickresidues"]:
            viewer.view.setStyle(
                {"resn": res, "byres": "true"},
                {"stick": {"colorscheme": "whiteCarbon"}},
            )

        surf = True
        surface_type = py3Dmol.MS
        match l_options["surfacetype"]:
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
                    "opacity": l_options["surfaceopacity"],
                    "color": l_options["surfacecolor"],
                },
                {"model": m_id},
            )

        if "note" in l_options:
            viewer._set_hover(self, m_id, l_options)
        return m_id

    def _repr_html_(self):
        # view = py3Dmol.view(
        #     width=self.draw_options["size"][0],
        #     height=self.draw_options["size"][1],
        #     options={"doAssembly": True},
        # )
        # self._viewer_add_(view)
        # view.zoomTo()
        # return view.write_html()
        return Viewer(  # noqa
            self,
            width=self.draw_options["size"][0],
            height=self.draw_options["size"][1],
        )._repr_html_()

    @property
    def viewer(self):
        """A viewer containing this receptor.

        Returns
        -------
        Viewer
        """
        return Viewer(
            self,
            width=self.draw_options["size"][0],
            height=self.draw_options["size"][1],
        )
