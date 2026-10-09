import numpy as np
from rdkit import Chem

from .._common import Mol
from ._base import ScoringFunction, _RDKitScoringFunction
from .dependencies import Dependency, PositionDependency


class InternalOverlap(ScoringFunction):
    """
    ✈️ — Calculates the overlap between atoms in a ligand.

    Uses the atom Van Der Waals radius as a measure of atom size. When atoms which have a
    topological distance ``> 4`` overlap, the overlap distance is added to the score. Every pair
    of atoms counts once, and only atoms in the molecule's ``scoring_mask`` take part (by default
    that leaves out the hydrogens, see :class:`~pyrite.Mol`).

    The score is computed from the atom positions alone, in numpy, for one pose or a whole batch
    at once. It does not depend on the rotation or translation, only on the torsions.

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
        self._first, self._second = np.nonzero(pairs)
        self._sum_of_radii = radii[self._first] + radii[self._second]

    def get_dependencies(self) -> list[Dependency]:
        return [self._position_dep]

    def _overlap(self, positions: np.ndarray) -> np.ndarray:
        """The summed overlap of ``(..., n_atoms, 3)`` positions: a number per pose."""
        offsets = positions[..., self._first, :] - positions[..., self._second, :]
        distances = np.sqrt(np.einsum("...pk,...pk->...p", offsets, offsets))
        return np.maximum(self._sum_of_radii - distances, 0.0).sum(axis=-1)

    def _score(self, pose, computed) -> float:
        return float(self._overlap(computed[self._position_dep]))

    def _batch_scores(self, poses, computed_batch) -> np.ndarray:
        return self._overlap(computed_batch[self._position_dep])

    def _score_and_gradient(self, pose, computed):
        # every overlapping pair adds (r_a + r_b - d): moving its atoms apart along their
        # connecting line lowers it by 1 per Angstrom
        positions = computed[self._position_dep]
        offsets = positions[self._first] - positions[self._second]
        distances = np.sqrt(np.einsum("pk,pk->p", offsets, offsets))
        overlapping = self._sum_of_radii > distances
        directions = offsets[overlapping] / distances[overlapping, None]
        forces = np.zeros_like(positions)
        np.add.at(forces, self._first[overlapping], -directions)
        np.add.at(forces, self._second[overlapping], directions)
        score = float((self._sum_of_radii[overlapping] - distances[overlapping]).sum())
        return score, self.mol.pose_gradient(pose, positions, forces)


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
