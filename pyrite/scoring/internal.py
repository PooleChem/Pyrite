import numpy as np
from rdkit import Chem

from .._common import Mol
from ._base import _RDKitScoringFunction


class InternalOverlap(_RDKitScoringFunction):
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
    molecule : Mol
        The molecule to be used for the calculation.
    ignore_hs : bool, default False
        If `ignore_hs` is ``True``, Hydrogen atoms will be masked out of the calculation.
    vdw_scale : float, default 1.0
        A multiplier of the Van Der Waals radii used for the calculation.

    See Also
    --------
    InternalEnergy
        More accurate, but slower approach.

    """

    def __init__(self, molecule: Mol, ignore_hs: bool = False, vdw_scale: float = 1.0):
        super().__init__(molecule)

        # Set up
        pt = Chem.GetPeriodicTable()
        a_nums = [atom.GetAtomicNum() for atom in self.mol.GetAtoms()]
        rvdw = np.array([pt.GetRvdw(z) * vdw_scale for z in a_nums], dtype=float)

        top_distance_matrix = Chem.GetDistanceMatrix(self.mol)

        self._sum_of_radii = rvdw[:, None] + rvdw[None, :]

        N = top_distance_matrix.shape[0]
        utri = np.triu(np.ones((N, N), dtype=bool), k=1)
        self._mask = utri & (top_distance_matrix > 4)

        if ignore_hs:
            self._mask &= (a_nums[:, None] > 1) & (a_nums[None, :] > 1)

    def _score(self, pose, computed) -> float:
        distance_matrix = Chem.Get3DDistanceMatrix(self.mol, computed[self.rdkit_dep])

        dists = distance_matrix[self._mask]
        r_sums = self._sum_of_radii[self._mask]
        overlaps = np.maximum(r_sums - dists, 0.0)

        return float(overlaps.sum())


class InternalEnergy(_RDKitScoringFunction):
    """
    🚗 — Calculates the MMFF internal energy of a molecule.

    Uses :mod:`rdkit` and the MMFF forcefield,
    using :func:`~rdkit.Chem.rdForceFieldHelpers.MMFFGetMoleculeForceField`.

    **Speed**: 🚗

    Parameters
    ----------
    molecule : Mol
        The molecule to be used for the calculation.

    See Also
    --------
    InternalOverlap
        Faster, but less accurate approach.

    """

    def __init__(self, molecule: Mol):
        super().__init__(molecule)

        mmff_props = Chem.AllChem.MMFFGetMoleculeProperties(self.mol)
        self._mmff_ff = Chem.AllChem.MMFFGetMoleculeForceField(self.mol, mmff_props)
        if self._mmff_ff is None:
            # MMFF94 cannot assign atom types for this molecule (unusual connectivity).
            # Fall back to UFF; if that also fails, internal energy returns 0.
            import warnings

            self._mmff_ff = Chem.AllChem.UFFGetMoleculeForceField(self.mol)
            if self._mmff_ff is None:
                warnings.warn(
                    f"InternalEnergy: MMFF and UFF both failed for {molecule}; "
                    "internal energy will be zero.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if self._mmff_ff is not None:
            self._mmff_ff.Initialize()

    def _score(self, pose, computed) -> float:
        if self._mmff_ff is None:
            return 0.0
        pos = self.mol.GetConformer(computed[self.rdkit_dep]).GetPositions()
        self._mmff_ff.Initialize()
        flat_pos = pos.reshape(-1).tolist()
        return self._mmff_ff.CalcEnergy(flat_pos)  # kcal/mol
