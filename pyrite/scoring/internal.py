import numpy as np
from numba import njit
from rdkit import Chem

from .._common import Mol
from ._base import ScoringFunction, _RDKitScoringFunction
from .dependencies import Dependency, PositionDependency


@njit(cache=True)
def _overlap_sum(positions, first, second, sum_of_radii):
    """The summed overlap of every pose: ``(n_poses, n_atoms, 3)`` positions -> ``(n_poses,)``."""
    n_poses = positions.shape[0]
    out = np.zeros(n_poses)
    for i in range(n_poses):
        total = 0.0
        for p in range(first.shape[0]):
            a, b = first[p], second[p]
            dx = positions[i, a, 0] - positions[i, b, 0]
            dy = positions[i, a, 1] - positions[i, b, 1]
            dz = positions[i, a, 2] - positions[i, b, 2]
            overlap = sum_of_radii[p] - np.sqrt(dx * dx + dy * dy + dz * dz)
            if overlap > 0.0:
                total += overlap
        out[i] = total
    return out


@njit(cache=True)
def _overlap_forces(positions, first, second, sum_of_radii):
    """The summed overlap of one pose, ``(n_atoms, 3)`` positions, and its gradient per atom.

    Every overlapping pair adds ``r_a + r_b - d``: moving its atoms apart along the line between
    them lowers it by 1 per Angstrom. Atoms at the same position have no direction, and no force.
    """
    forces = np.zeros_like(positions)
    total = 0.0
    for p in range(first.shape[0]):
        a, b = first[p], second[p]
        dx = positions[a, 0] - positions[b, 0]
        dy = positions[a, 1] - positions[b, 1]
        dz = positions[a, 2] - positions[b, 2]
        distance = np.sqrt(dx * dx + dy * dy + dz * dz)
        overlap = sum_of_radii[p] - distance
        if overlap <= 0.0:
            continue
        total += overlap
        if distance > 0.0:
            ux, uy, uz = dx / distance, dy / distance, dz / distance
            forces[a, 0] -= ux
            forces[a, 1] -= uy
            forces[a, 2] -= uz
            forces[b, 0] += ux
            forces[b, 1] += uy
            forces[b, 2] += uz
    return total, forces


class InternalOverlap(ScoringFunction):
    """
    ✈️ — Calculates the overlap between atoms in a ligand.

    Uses the atom Van Der Waals radius as a measure of atom size. When atoms which have a
    topological distance ``> 4`` overlap, the overlap distance is added to the score. Every pair
    of atoms counts once, and only atoms in the molecule's ``scoring_mask`` take part (by default
    that leaves out the hydrogens, see :class:`~pyrite.Mol`).

    The score is computed from the atom positions alone, in one compiled loop over the pairs of
    atoms, for one pose or a whole batch at once. It does not depend on the rotation or
    translation, only on the torsions.

    .. note::
        This class can be used as a measure of internal ligand energy. However, it does not fully
        take torsion angles into account. For a more accurate, albeit slower, method to calculate
        internal energy, see :class:`InternalEnergy`.

    **Speed**: ✈️

    Parameters
    ----------
    molecule : Mol
        The molecule to be used for the calculation.
    vdw_scale : float, default 1.0
        A multiplier of the Van Der Waals radii used for the calculation.

    See Also
    --------
    InternalEnergy
        More accurate, but slower approach.

    Examples
    --------
    >>> from pyrite.scoring import InternalOverlap
    >>> clashes = 0.5 * InternalOverlap(ligand)
    """

    def __init__(self, molecule: Mol, vdw_scale: float = 1.0):
        self.mol = molecule
        self._position_dep = PositionDependency(molecule)

        periodic_table = Chem.GetPeriodicTable()
        atomic_numbers = np.array([atom.GetAtomicNum() for atom in molecule.atoms])
        radii = np.array([periodic_table.GetRvdw(int(z)) * vdw_scale for z in atomic_numbers])

        topological_distance = np.asarray(Chem.GetDistanceMatrix(molecule.rdkit))
        pairs = np.triu(np.ones(topological_distance.shape, dtype=bool), k=1)
        pairs &= topological_distance > 4
        counted = np.asarray(molecule.scoring_mask)
        pairs &= counted[:, None] & counted[None, :]

        # the atoms of every counted pair, and the sum of their radii
        self._first, self._second = (
            np.ascontiguousarray(a, dtype=np.int64) for a in np.nonzero(pairs)
        )
        self._sum_of_radii = np.ascontiguousarray(radii[self._first] + radii[self._second])

    def get_dependencies(self) -> list[Dependency]:
        return [self._position_dep]

    def _overlap(self, positions: np.ndarray) -> np.ndarray:
        """The summed overlap of ``(n_poses, n_atoms, 3)`` positions: a number per pose."""
        positions = np.ascontiguousarray(positions, dtype=np.float64)
        return _overlap_sum(positions, self._first, self._second, self._sum_of_radii)

    def _score(self, pose, computed) -> float:
        return float(self._overlap(computed[self._position_dep][None])[0])

    def _batch_scores(self, poses, computed_batch) -> np.ndarray:
        return self._overlap(computed_batch[self._position_dep])

    def _score_and_gradient(self, pose, computed):
        positions = computed[self._position_dep]
        score, forces = _overlap_forces(
            np.ascontiguousarray(positions, dtype=np.float64),
            self._first,
            self._second,
            self._sum_of_radii,
        )
        return float(score), self.mol.pose_gradient(pose, positions, forces)


class InternalEnergy(_RDKitScoringFunction):
    """
    🚗 — Calculates the MMFF internal energy of a molecule.

    Uses :mod:`rdkit` and the MMFF forcefield,
    using `MMFFGetMoleculeForceField <https://www.rdkit.org/docs/source/rdkit.Chem.rdForceFieldHelpers.html>`_.

    MMFF needs every hydrogen. When the molecule does not have them all (``hydrogens="polar"``,
    the default, or ``"remove"``), the missing ones are added to each pose at ideal positions
    computed from the heavy atoms (`AddHs <https://www.rdkit.org/docs/source/rdkit.Chem.rdmolops.html>`_ with ``addCoords``), and the
    energy is that of the complete molecule. A molecule that has all its hydrogens is scored as
    it is.

    **Speed**: 🚗

    Parameters
    ----------
    molecule : Mol
        The molecule to be used for the calculation.

    See Also
    --------
    InternalOverlap
        Faster, but less accurate approach.

    Examples
    --------
    >>> from pyrite.scoring import InternalEnergy
    >>> ligand = Mol.from_sdf("ligand.sdf", flexible=True, hydrogens="keep")
    >>> strain = 1e-2 * InternalEnergy(ligand)
    """

    def __init__(self, molecule: Mol):
        super().__init__(molecule)

        complete = self._with_all_hydrogens(self.mol.rdkit)
        # The force field is built once, on the complete molecule; a pose only changes positions.
        self._n_atoms = complete.GetNumAtoms()
        self._adds_hydrogens = self._n_atoms > self.mol.n_atoms

        mmff_props = Chem.AllChem.MMFFGetMoleculeProperties(complete)
        self._mmff_ff = Chem.AllChem.MMFFGetMoleculeForceField(complete, mmff_props)
        if self._mmff_ff is None:
            # MMFF94 cannot assign atom types for this molecule (unusual connectivity).
            # Fall back to UFF; if that also fails, internal energy returns 0.
            import warnings

            self._mmff_ff = Chem.AllChem.UFFGetMoleculeForceField(complete)
            if self._mmff_ff is None:
                warnings.warn(
                    f"InternalEnergy: MMFF and UFF both failed for {molecule}; "
                    "internal energy will be zero.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if self._mmff_ff is not None:
            self._mmff_ff.Initialize()

    @staticmethod
    def _with_all_hydrogens(rdkit_mol: Chem.Mol) -> Chem.Mol:
        """Return `rdkit_mol` with its implicit hydrogens added, placed from the heavy atoms.

        The added hydrogens come after the existing atoms, so those keep their indices.
        """
        rdkit_mol = Chem.Mol(rdkit_mol)
        rdkit_mol.UpdatePropertyCache(strict=False)
        return Chem.AddHs(rdkit_mol, addCoords=True)

    def _score(self, pose, computed) -> float:
        if self._mmff_ff is None:
            return 0.0
        posed = computed[self.rdkit_dep]
        if self._adds_hydrogens:
            posed = self._with_all_hydrogens(posed)
            if posed.GetNumAtoms() != self._n_atoms:
                raise RuntimeError("InternalEnergy: adding hydrogens to the pose gave other atoms.")
        pos = posed.GetConformer().GetPositions()
        self._mmff_ff.Initialize()
        flat_pos = pos.reshape(-1).tolist()
        return self._mmff_ff.CalcEnergy(flat_pos)  # kcal/mol
