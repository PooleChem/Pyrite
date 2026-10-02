import numpy as np
from rdkit import Chem
from scipy.spatial import cKDTree

from .._common import AtomType, Mol
from .._util import _symmetry_mappings
from ._base import ScoringFunction, _RDKitScoringFunction
from .dependencies import Dependency, PositionDependency


class RMSD(_RDKitScoringFunction):
    """
    🚗 — Used to calculate the RMSD between two poses.

    The RMSD is calculated using :func:`~rdkit.Chem.rdMolAlign.CalcRMS`.

    **Speed**: 🚲–🚄, depending on the size of the ligand.

    Parameters
    ----------
    probe_mol : Mol
        The ligand to be used for the calculation. This is the probe molecule, i.e., the molecule
        whose pose changes.
    ref_mol : Mol, optional
        The stationary ligand to be used for the calculation. The conformation of this ligand
        should not change. If not provided, a copy of `ligand` will be used.
    max_matches : int, default 1000
        The maximum number of symmetry-equivalent atom mappings to consider. The symmetry of the
        molecule is taken into account: the RMSD is the lowest over all mappings, so a pose with
        a flipped phenyl ring is not far from the pose it is a flip of. A warning is issued when
        the limit is reached, as the RMSD can then be overestimated.

    """

    def __init__(self, probe_mol: Mol, ref_mol: Mol = None, max_matches: int = 1000):
        super().__init__(probe_mol)
        self.probe_mol = probe_mol
        same_atom_order = ref_mol is None
        if ref_mol is None:
            self.ref_mol = probe_mol.copy()
        else:
            self.ref_mol = ref_mol

        matches = _symmetry_mappings(
            self.probe_mol, self.ref_mol, max_matches, include_identity=same_atom_order
        )
        self._atom_map = [
            [(probe_atom, ref_atom) for ref_atom, probe_atom in enumerate(m)] for m in matches
        ]

    def _score(self, pose, computed) -> float:
        score = Chem.rdMolAlign.CalcRMS(
            computed[self.rdkit_dep],
            self.ref_mol.rdkit,
            map=self._atom_map,
        )
        return score


class Crowding(_RDKitScoringFunction):
    r"""
    🚲 — Used to determine the similarity of a pose to a set of poses.

    New poses can be registered using :meth:`register_pose`. When :meth:`get_score` is called,
    the RMSD of the :class:`~pyrite.Mol` pose to each registered pose will be calculated using
    :func:`rdkit.Chem.rdMolAlign.CalcRMS`.

    The score is calculated using the following formula:

    .. math::
        score = \sum_{i} 4^{(rmsd(i) - offset)}

    If `divide` is ``True``, the score is then divided by the number of poses.

    The function for every pose is then shown below, where in this example the `offset` is 4.0.

    .. plot::
       :width: 80%
       :alt: Crowding example

       import numpy as np
       import matplotlib.pyplot as plt

       offset = 4.0

       x = np.linspace(0, 8, 400)
       y = 4 ** (-x + offset)

       plt.figure()
       plt.plot(x, y, linewidth=2)
       plt.xlabel("RMSD")
       plt.ylabel("Score")

    **Speed**: 🐢–🚄, depending on the size of the mol and the number of registered poses.

    Parameters
    ----------
    mol : Mol
        The mol to be used for the calculation. This is the probe molecule, i.e., the molecule
        whose pose changes.
    offset : float, default 4.0
        The offset used in the exponential score calculation.
    register_initial : bool, default False
        If `register_initial` is ``True``, the initial pose will be registered as well.
    divide : bool, default True
        If `divide` is ``True``, the score will be divided by the number of registered poses.
    max_matches : int, default 1000
        The maximum number of symmetry-equivalent atom mappings to consider, see :class:`RMSD`.

    """

    def __init__(
        self,
        molecule: Mol,
        offset: float = 4.0,
        register_initial: bool = False,
        divide: bool = True,
        max_matches: int = 1000,
    ):
        super().__init__(molecule)

        self._ref_mol = molecule.copy()  # holds the registered poses as its conformers

        self._registered_conf = []
        if register_initial:
            self._registered_conf.append(0)

        self._offset = offset
        self._divide = divide

        matches = _symmetry_mappings(self.mol, self.mol, max_matches, include_identity=True)
        self._atom_map = [
            [(probe_atom, ref_atom) for ref_atom, probe_atom in enumerate(m)] for m in matches
        ]

    def register_pose(self, v):
        """Register a new :class:`~pyrite.Mol` pose.

        Parameters
        ----------
        v : array_like, int
            Either a list containing the variables used to create the new pose using
            :meth:`~pyrite.Mol.pose_to_conformer`, or a `conf_id`.

        """
        if not isinstance(v, int):
            conf_id = self._ref_mol.pose_to_conformer(v, new_conf=True)
        else:
            conf_id = v
        self._registered_conf.append(conf_id)

    def _score(self, pose, computed) -> float:
        posed = computed[self.rdkit_dep]
        score = 0

        for i in self._registered_conf:
            rms = Chem.rdMolAlign.CalcRMS(
                posed,
                self._ref_mol.rdkit,
                refId=i,
                map=self._atom_map,
            )

            score += 4 ** (-rms + self._offset)

        if self._divide and len(self._registered_conf) > 0:
            score /= len(self._registered_conf)

        return score


class NumProteinAtomsWithinA(ScoringFunction):
    """
    🚲 — Returns the total number of protein atoms within ``A`` Angstroms of the ligand atoms.

    The search is executed using a :class:`~scipy.spatial.KDTree`, on each atom of the ligand.
    Therefore, the same protein atom can be counted multiple times.

    **Speed**: 🚶 -- 🚄, depending on ``A`` and the size of the ligand.

    Parameters
    ----------
    probe_mol : Mol
        The ligand to be used for the calculation.
    ref_mol : Mol
        The receptor to be used for the calculation.
    a : float, default 4.0
        The radius (in Angstroms) within which to search for protein atoms.
    ignore_hs_ligand : bool, default False
        If `ignore_hs_ligand` is ``True``, Hydrogen atoms will be masked out of the calculation.

    """

    def __init__(
        self,
        probe_mol: Mol,
        ref_mol: Mol,
        a: float = 4.0,
        ignore_hs_ligand: bool = False,
    ):
        self.probe_mol = probe_mol
        self._mask = np.array(
            [not ignore_hs_ligand or atom.GetAtomicNum() > 1 for atom in self.probe_mol.atoms]
        )
        self.ref_mol = ref_mol
        self.a = a
        self.tree = cKDTree(self.ref_mol.positions)
        self._position_dep = PositionDependency(self.probe_mol)

    def get_dependencies(self) -> list[Dependency]:
        return [self._position_dep]

    def _score(self, pose, computed) -> float:
        conf_pos = computed[self._position_dep]

        return sum(self.tree.query_ball_point(conf_pos[self._mask], self.a, return_length=True))


class NumTors(ScoringFunction):
    """
     🚀 — Returns the number of torsions (torsion angles) in the ligand.

    **Speed**: 🚀

    Parameters
    ----------
    molecule : Mol
        The ligand to be used.

    """

    def __init__(self, molecule: Mol):
        self.mol = molecule

        self.result = len(self.mol.rotatable_torsions)

    def _score(self, *args, **kwargs):
        return self.result


class NumAtoms(ScoringFunction):
    """
     🚀 — Returns the number of atoms in the ligand.

    **Speed**: 🚀

    Parameters
    ----------
    molecule : Mol
        The ligand to be used.
    include_hs : bool, default False
        If `include_hs` is ``False``, only heavy atoms will be counted.
    """

    def __init__(self, molecule: Mol, include_hs: bool = False):
        self.mol = molecule
        self.include_hs = include_hs

        mask = np.ones(len(self.mol.positions), dtype=bool)
        if not self.include_hs:
            mask &= self.mol.atom_types != AtomType.Hydrogen
            mask &= self.mol.atom_types != AtomType.PolarHydrogen
        self.result = np.sum(mask)

    def _score(self, *args, **kwargs):
        return self.result
