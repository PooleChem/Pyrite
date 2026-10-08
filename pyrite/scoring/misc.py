import numpy as np
from rdkit import Chem
from scipy.spatial import cKDTree

from .._common import Mol
from .._util import _heavy_atoms, _symmetry_mappings
from ._base import ScoringFunction, _RDKitScoringFunction
from .dependencies import Dependency, PositionDependency


class RMSD(_RDKitScoringFunction):
    """
    🚗 — Used to calculate the RMSD between two poses.

    The RMSD is calculated using `CalcRMS <https://www.rdkit.org/docs/source/rdkit.Chem.rdMolAlign.html>`_, over the heavy atoms
    only (the convention for docking poses): the hydrogens the molecule has, which depend on how it
    was loaded, do not change it. The poses are compared in place, without aligning them.

    **Speed**: 🚗, depending on the size and the symmetry of the ligand.

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

    See Also
    --------
    pyrite.cluster.rmsd_matrix : The RMSD between all pairs of poses.
    pyrite.Mol.input_pose : The pose of the input geometry.

    Examples
    --------
    >>> from pyrite.scoring import RMSD
    >>> ligand = Mol.from_sdf("ligand.sdf", flexible=True)  # the crystal pose
    >>> rmsd = RMSD(ligand)
    >>> rmsds = rmsd.batch_scores(docked_poses)
    >>> success = rmsds.min() < 2.0
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
            self.probe_mol,
            self.ref_mol,
            max_matches,
            include_identity=same_atom_order,
            heavy_atoms=True,
        )
        ref_atoms = _heavy_atoms(self.ref_mol)
        self._atom_map = [
            [(probe_atom, int(ref_atoms[k])) for k, probe_atom in enumerate(m)] for m in matches
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
    🐢 — Used to determine the similarity of a pose to a set of poses.

    New poses can be registered using :meth:`register_pose`. When :meth:`get_score` is called,
    the RMSD of the :class:`~pyrite.Mol` pose to each registered pose will be calculated using
    `CalcRMS <https://www.rdkit.org/docs/source/rdkit.Chem.rdMolAlign.html>`_.

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

    **Speed**: 🐢 with one registered pose; the time grows with every registered pose.

    Parameters
    ----------
    molecule : Mol
        The molecule to be used for the calculation. This is the probe molecule, i.e., the
        molecule whose pose changes.
    offset : float, default 4.0
        The offset used in the exponential score calculation.
    register_initial : bool, default False
        If `register_initial` is ``True``, the initial pose will be registered as well.
    divide : bool, default True
        If `divide` is ``True``, the score will be divided by the number of registered poses.
    max_matches : int, default 1000
        The maximum number of symmetry-equivalent atom mappings to consider, see :class:`RMSD`.

    See Also
    --------
    RMSD : The RMSD to one reference pose.

    Examples
    --------
    >>> from pyrite.scoring import Crowding
    >>> crowding = Crowding(ligand, offset=4.0)
    >>> crowding.register_pose(previous_best)
    >>> diverse = scoring_function + crowding
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

        Every registered pose adds a term to the score, so a search is pushed away from poses that
        were already found.

        Parameters
        ----------
        v : Pose, array_like or int
            A pose (or its values), which is put on a new conformer of the reference copy with
            :meth:`~pyrite.Mol.pose_to_conformer`, or the id of a conformer of that copy. Pass a
            conformer id as a Python ``int``: a numpy integer is taken as a pose.

        See Also
        --------
        Crowding : The scoring function.

        Examples
        --------
        >>> crowding.register_pose(result.x)
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
    🚗 — Returns the total number of protein atoms within ``A`` Angstroms of the ligand atoms.

    The search is executed using a :class:`~scipy.spatial.KDTree`, on each atom of the ligand.
    Therefore, the same protein atom can be counted multiple times.

    Only the atoms in the ``scoring_mask`` of the two molecules take part (by default the hydrogens
    are left out, so the count does not depend on whether the files contain them).

    **Speed**: 🚗, depending on ``A`` and the size of the ligand.

    Parameters
    ----------
    probe_mol : Mol
        The ligand to be used for the calculation.
    ref_mol : Mol
        The receptor to be used for the calculation.
    a : float, default 4.0
        The radius (in Angstroms) within which to search for protein atoms.

    See Also
    --------
    pyrite.scoring.dependencies.KNNDependency : The nearest neighbor search of the KNN-based terms.

    Examples
    --------
    >>> from pyrite.scoring import NumProteinAtomsWithinA
    >>> contacts = NumProteinAtomsWithinA(ligand, receptor, a=4.0)
    """

    def __init__(
        self,
        probe_mol: Mol,
        ref_mol: Mol,
        a: float = 4.0,
    ):
        self.probe_mol = probe_mol
        self._mask = np.asarray(probe_mol.scoring_mask)
        self.ref_mol = ref_mol
        self.a = a
        self.tree = cKDTree(self.ref_mol.positions[np.asarray(ref_mol.scoring_mask)])
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

    See Also
    --------
    NumAtoms : The number of atoms.
    pyrite.Mol.n_tors : The number of torsions of a molecule.

    Examples
    --------
    >>> from pyrite.scoring import NumTors
    >>> vina_like = receptor_terms / (1 + 0.0585 * NumTors(ligand))  # Vina's torsion penalty
    """

    def __init__(self, molecule: Mol):
        self.mol = molecule

        self.result = len(self.mol.rotatable_torsions)

    def _score(self, *args, **kwargs):
        return self.result

    def _score_and_gradient(self, pose, computed):
        # the number of torsions does not depend on the pose
        return self.result, np.zeros(len(np.asarray(pose)))


class NumAtoms(ScoringFunction):
    """
    🚀 — Returns the number of atoms in the ligand.

    Only the atoms in the molecule's ``scoring_mask`` are counted: by default the heavy atoms.

    **Speed**: 🚀

    Parameters
    ----------
    molecule : Mol
        The ligand to be used.

    See Also
    --------
    NumTors : The number of torsions.

    Examples
    --------
    >>> from pyrite.scoring import NumAtoms
    >>> per_atom = scoring_function / NumAtoms(ligand)
    """

    def __init__(self, molecule: Mol):
        self.mol = molecule
        self.result = int(np.sum(molecule.scoring_mask))

    def _score(self, *args, **kwargs):
        return self.result
