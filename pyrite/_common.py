from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Literal

import numpy as np
import py3Dmol
from IPython.display import SVG, Image
from numpy.typing import NDArray
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, SDWriter

from .view import Viewer
from .atom_consts import AtomType, vina_atom_consts

if TYPE_CHECKING:
    from pyrite.bounds import Bounds


def _rotation_matrix_to_euler(r: NDArray):
    """
    Extract ZYX (yaw-pitch-roll) Euler angles from a 3×3 rotation matrix R.
    Returns (phi, theta, psi) = (roll, pitch, yaw) in radians.
    """
    # # clamp to handle numerical errors outside [-1,1]
    sy = -r[2, 0]
    theta = np.arcsin(np.clip(sy, -1.0, 1.0))

    # Check for gimbal lock
    if np.isclose(np.cos(theta), 0.0):
        # Gimbal lock: pitch is ±90°
        # Roll and yaw are coupled; set roll=0 and compute yaw:
        phi = 0.0
        psi = np.arctan2(-r[0, 1], r[1, 1])
    else:
        phi = np.arctan2(r[2, 1], r[2, 2])  # roll
        psi = np.arctan2(r[1, 0], r[0, 0])  # yaw

    return phi, theta, psi


# TODO: Support Quaternions
def _rotation_matrix_from_euler(roll: float, pitch: float, yaw: float) -> NDArray:
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


def _translation_matrix_from_coordinates(x: float, y: float, z: float) -> NDArray:
    """Create a translation matrix from coordinates.

    Create a 4x4 translation matrix from the provided relative x, y, and z coordinates.

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


PROTEINS_HEAVY_ATOMS_CUTOFF = 1000


class Mol(Chem.Mol):
    """
    Representation of a molecule.

    The Mol class provides methods for initializing molecules from various sources,
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
        center_atom: int = None,
        flex_hydrogens: bool = False,
        flexible: bool = False,  # TODO: allow list of resids. No. Read everything rigid, and allow for auto setting of dihedrals, or manual, or by resid.
    ):
        self.__cur_transform = np.eye(4)
        self.__cur_rotation = [0, 0, 0]
        self.__rotatable_dihedrals = np.array([], dtype=object)
        self.__dihedral_angles = np.array([])

        self._fix_mol_valence(sanitize=False)  # TODO: sanitize?

        if hydrogens == "add":
            mol = Chem.AllChem.AddHs(mol, addCoords=True, sanitize=False)
        elif hydrogens == "remove":
            mol = Chem.AllChem.RemoveHs(mol, sanitize=False)

        # TODO: keep for prot?
        self._center_atom = (
            center_atom if center_atom is not None else self.__get_center_atom()
        )

        if self.GetNumConformers() == 0:
            Chem.SanitizeMol(self)
            params = Chem.AllChem.ETKDGv3()
            params.randomSeed = 0xC0FFEE
            Chem.AllChem.EmbedMolecule(self, params)  # TODO: cant do if not sanitized.

        self.__flexible = flexible
        if flexible:
            self.__compute_rotatable_dihedrals(flex_hydrogens)

        self.__assign_atom_types()
        Chem.rdPartialCharges.ComputeGasteigerCharges(self)


        # TODO: keep?
        self._init_state = (
            copy.deepcopy(self.__cur_rotation),
            copy.deepcopy(self.position),
            copy.deepcopy(self.__dihedral_angles),
        )

        # If more than 1000 heavy atoms: assume protein for drawing
        if self.GetNumHeavyAtoms() >= PROTEINS_HEAVY_ATOMS_CUTOFF:
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

        mol = Chem.MolFromPDBFile(
            pdb_file, sanitize=False, removeHs=(hydrogens == "remove")
        )

        if template_smiles or template_sdf:
            if template_smiles:
                template_mol = Chem.MolFromSmiles(template_smiles)
            else:
                template_mol = Chem.MolFromMolFile(template_sdf)

            mol = Chem.AllChem.AssignBondOrdersFromTemplate(template_mol, mol)

            Chem.AssignStereochemistryFrom3D(mol)
            Chem.SanitizeMol(mol)

        return cls(mol, hydrogens=hydrogens, **kwargs)

    # TODO: be able to load and return multiple ligands from the same SDF file. (for v_from_sdf too)
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
            The conformations in the SDF file, represented by a tuple of size ``(6 + n_dihedrals)``,
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
        with Chem.SDMolSupplier(
            mol_file, removeHs=(hydrogens == "remove"), sanitize=False
        ) as supl:
            for pose in supl:
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

    def _fix_mol_valence(self, sanitize=True):
        Chem.SanitizeMol(
            self,
            sanitizeOps=(
                Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
            ),
        )

        for atom in self.GetAtoms():
            # print(
            #     atom.GetSymbol(),
            #     atom.GetValence(Chem.ValenceType.EXPLICIT),
            #     atom.GetFormalCharge(),
            # )

            if (
                atom.GetSymbol() == "N"
                and atom.GetValence(Chem.ValenceType.EXPLICIT) == 4
            ):
                atom.SetFormalCharge(+1)
            if (
                atom.GetSymbol() == "O"
                and atom.GetValence(Chem.ValenceType.EXPLICIT) == 1
            ):
                atom.SetFormalCharge(-1)
            # incorrect epoxide fix TODO: check with David
            if (
                atom.GetSymbol() == "O"
                and atom.GetValence(Chem.ValenceType.EXPLICIT) == 2
            ):
                atom.SetFormalCharge(0)

        self.UpdatePropertyCache(strict=sanitize)

        if sanitize:
            Chem.SanitizeMol(self)
        return self

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
            else self.GetNumHeavyAtoms() >= PROTEINS_HEAVY_ATOMS_CUTOFF
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

    def atom_type(self, atom_index: int, mask: NDArray = None):
        """Get the atom type of the specified atom, optionally taking `mask` into account.
        # TODO: what does that mean? Revise receptor masking?

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

    # TODO: cache positions
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

    # TODO: optimizations here would be great (even with loss of default conformer updating?)
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

        rotate = _rotation_matrix_from_euler(roll, pitch, yaw)
        translate = _translation_matrix_from_coordinates(x, y, z)

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
        assert len(new_vars) == 6 + len(self.__rotatable_dihedrals), f'Unexpected number of variables: {len(new_vars)}. Expected: {6 + len(self.__rotatable_dihedrals)}.'

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
        if conformations not in {
            "conformer",
            "random",
        }:  # TODO: add none (just default dihedrals)
            raise ValueError("conformations must be either 'conformer' or 'random'")
        if combine not in {"random", "grid"}:  # TODO: rename to product
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
        """Write the current molecule with positions `v` to an SDF file.

        Parameters
        ----------
        file : str
            The path to the file to create.
        v : array_like
            An array of shape ``(6 + n_dihedrals, n)``, containing molecular positions to write.

        """

        writer = Chem.SDWriter(file)

        for var in v:
            conf_id = self.update(var, new_conf=True)
            self.to_sdf(writer, conf_id=conf_id)
            self.RemoveConformer(conf_id)
        writer.close()

    def __hash__(self):
        # TODO!
        return id(self)

