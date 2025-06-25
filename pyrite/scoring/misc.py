import numpy as np
from rdkit import Chem
from scipy.spatial import cKDTree

from ._base import ScoringFunction
from .._common import Ligand, Receptor, AtomType


class RMSD(ScoringFunction):
    """
    🚗 — Used to calculate the RMSD between two poses.

    The RMSD is calculated using :func:`~rdkit.Chem.rdMolAlign.CalcRMS`.

    **Speed**: 🚲–🚄, depending on the size of the ligand.

    Parameters
    ----------
    ligand : Ligand
        The ligand to be used for the calculation. This is the probe molecule, i.e., the molecule
        whose pose changes.
    crystal_ligand : Ligand, optional
        The stationary ligand to be used for the calculation. The conformation of this ligand
        should not change. If not provided, a copy of `ligand` will be used.

    """

    def __init__(self, ligand: Ligand, crystal_ligand: Ligand = None):
        self.ligand = ligand
        if crystal_ligand is None:
            self.crystal_ligand = type(ligand)(Chem.Mol(ligand))
        else:
            self.crystal_ligand = crystal_ligand

        matches = self.ligand.GetSubstructMatches(
            self.crystal_ligand, uniquify=True, useChirality=True
        )
        self._atom_map = [
            list(zip(range(self.ligand.GetNumAtoms()), match)) for match in matches
        ]

    def _score(self, conf_id, *args, **kwargs) -> float:
        score = Chem.rdMolAlign.CalcRMS(
            self.ligand,
            self.crystal_ligand,
            prbId=conf_id,
            map=self._atom_map,
        )
        return score


class Crowding(ScoringFunction):
    r"""
    🚲 — Used to determine the similarity of a pose to a set of poses.

    New poses can be registered using :meth:`register_pose`. When :meth:`get_score` is called,
    the RMSD of the :class:`~pyrite.Ligand` pose to each registered pose will be calculated using
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

    **Speed**: 🐢–🚄, depending on the size of the ligand and the number of registered poses.

    Parameters
    ----------
    ligand : Ligand
        The ligand to be used for the calculation. This is the probe molecule, i.e., the molecule
        whose pose changes.
    offset : float, default 4.0
        The offset used in the exponential score calculation.
    register_initial : bool, default False
        If `register_initial` is ``True``, the initial pose will be registered as well.
    divide : bool, default True
        If `divide` is ``True``, the score will be divided by the number of registered poses.

    """

    def __init__(
        self,
        ligand: Ligand,
        offset: float = 4.0,
        register_initial: bool = False,
        divide: bool = True,
    ):
        self.ligand = ligand

        self._ref_ligand = type(ligand)(Chem.Mol(ligand))

        self._registered_conf = []
        if register_initial:
            self._registered_conf.append(0)

        self._offset = offset
        self._divide = divide

        matches = self.ligand.GetSubstructMatches(
            self.ligand, uniquify=True, useChirality=True
        )
        self._atom_map = [
            list(zip(range(self.ligand.GetNumAtoms()), match)) for match in matches
        ]

    def register_pose(self, v):
        """Register a new :class:`~pyrite.Ligand` pose.

        Parameters
        ----------
        v : array_like, int
            Either a list containing the variables used to create the new pose using
            :meth:`~pyrite.Ligand.update`, or a `conf_id`.

        """
        if not isinstance(v, int):
            conf_id = self._ref_ligand.update(v, new_conf=True)
        else:
            conf_id = v
        self._registered_conf.append(conf_id)

    def _score(self, conf_id, *args, **kwargs) -> float:
        score = 0

        for i in self._registered_conf:
            rms = Chem.rdMolAlign.CalcRMS(
                self.ligand,
                self._ref_ligand,
                prbId=conf_id,
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
    ligand : Ligand
        The ligand to be used for the calculation.
    receptor : Receptor
        The receptor to be used for the calculation.
    a : float, default 4.0
        The radius (in Angstroms) within which to search for protein atoms.
    ignore_hs_ligand : bool, default False
        If `ignore_hs_ligand` is ``True``, Hydrogen atoms will be masked out of the calculation.

    """

    def __init__(
        self,
        ligand: Ligand,
        receptor: Receptor,
        a: float = 4.0,
        ignore_hs_ligand: bool = False,
    ):
        self.ligand = ligand
        self._mask = np.array(
            [
                not ignore_hs_ligand or atom.GetAtomicNum() > 1
                for atom in self.ligand.GetAtoms()
            ]
        )
        self.receptor = receptor
        self.a = a
        self.tree = cKDTree(self.receptor.positions)

    def _score(self, conf_id, computed) -> float:
        conf_pos = self.ligand.GetConformer(conf_id).GetPositions()

        return sum(
            self.tree.query_ball_point(conf_pos[self._mask], self.a, return_length=True)
        )


class NumTors(ScoringFunction):
    """
     🚀 — Returns the number of torsions (dihedral angles) in the ligand.

    **Speed**: 🚀

    Parameters
    ----------
    ligand : Ligand
        The ligand to be used.

    """

    def __init__(self, ligand: Ligand):
        self.ligand = ligand

        self.result = len(self.ligand.rotatable_dihedrals)

    def _score(self, *args, **kwargs):
        return self.result


class NumAtoms(ScoringFunction):
    """
     🚀 — Returns the number of atoms in the ligand.

    **Speed**: 🚀

    Parameters
    ----------
    ligand : Ligand
        The ligand to be used.
    include_hs : bool, default False
        If `include_hs` is ``False``, only heavy atoms will be counted.
    """

    def __init__(self, ligand: Ligand, include_hs: bool = False):
        self.ligand = ligand
        self.include_hs = include_hs

        mask = np.ones(len(self.ligand.positions), dtype=bool)
        if not self.include_hs:
            mask &= self.ligand.atom_types != AtomType.Hydrogen
            mask &= self.ligand.atom_types != AtomType.PolarHydrogen
        self.result = np.sum(mask)

    def _score(self, *args, **kwargs):
        return self.result
