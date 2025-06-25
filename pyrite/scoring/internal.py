from rdkit import Chem

import numpy as np

from ._base import ScoringFunction
from .._common import Ligand


class InternalOverlap(ScoringFunction):
    """
    ✈️ — Calculates the overlap between atoms in a ligand.

    Uses the atom Van Der Waals radius as a measure of atom size. When atoms which have a
    topological distance ``> 4`` overlap, the overlap distance is added to the score.

    .. note::
        This class can be used as a measure of internal ligand energy. However, it does not fully
        take torsion angles into account. For a more accurate, albeit slower, method to calculate
        internal energy, see :class:`InternalEnergy`.

    **Speed**: ✈️

    Parameters
    ----------
    ligand : Ligand
        The ligand to be used for the calculation.
    ignore_hs : bool, default False
        If `ignore_hs` is ``True``, Hydrogen atoms will be masked out of the calculation.
    vdw_scale : float, default 1.0
        A multiplier of the Van Der Waals radii used for the calculation.

    See Also
    --------
    InternalEnergy
        More accurate, but slower approach.

    """

    def __init__(self, ligand: Ligand, ignore_hs: bool = False, vdw_scale: float = 1.0):
        self.ligand = ligand

        # Set up
        pt = Chem.GetPeriodicTable()
        a_nums = [atom.GetAtomicNum() for atom in self.ligand.GetAtoms()]
        rvdw = np.array([pt.GetRvdw(z) * vdw_scale for z in a_nums], dtype=float)

        top_distance_matrix = Chem.GetDistanceMatrix(self.ligand)

        self._sum_of_radii = rvdw[:, None] + rvdw[None, :]

        N = top_distance_matrix.shape[0]
        utri = np.triu(np.ones((N, N), dtype=bool), k=1)
        self._mask = utri & (top_distance_matrix > 4)

        if ignore_hs:
            self._mask &= (a_nums[:, None] > 1) & (a_nums[None, :] > 1)

    def _score(self, conf_id, *args, **kwargs) -> float:
        distance_matrix = Chem.Get3DDistanceMatrix(self.ligand, conf_id)

        dists = distance_matrix[self._mask]
        r_sums = self._sum_of_radii[self._mask]
        overlaps = np.maximum(r_sums - dists, 0.0)

        return float(overlaps.sum())


class InternalEnergy(ScoringFunction):
    """
    🚗 — Calculates the MMFF internal energy of a ligand.

    Uses :mod:`rdkit` and the MMFF forcefield,
    using :func:`~rdkit.Chem.rdForceFieldHelpers.MMFFGetMoleculeForceField`.

    **Speed**: 🚗

    Parameters
    ----------
    ligand : Ligand
        The ligand to be used for the calculation.

    See Also
    --------
    InternalOverlap
        Faster, but less accurate approach.

    """

    def __init__(self, ligand: Ligand):
        self.ligand = ligand

        mmff_props = Chem.AllChem.MMFFGetMoleculeProperties(self.ligand)
        self._mmff_ff = Chem.AllChem.MMFFGetMoleculeForceField(self.ligand, mmff_props)
        self._mmff_ff.Initialize()

    def _score(self, conf_id, *args, **kwargs) -> float:
        pos = self.ligand.GetConformer(conf_id).GetPositions()
        self._mmff_ff.Initialize()
        flat_pos = pos.reshape(-1).tolist()
        return self._mmff_ff.CalcEnergy(flat_pos)  # kcal/mol
