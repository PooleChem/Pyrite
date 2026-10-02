import numpy as np

from .._common import Mol
from ..bounds import Bounds, Pocket
from ._base import ScoringFunction
from .dependencies import Dependency, KNNDependency, PositionDependency, PositionQuery

# pylint: disable=too-few-public-methods


class DistanceToPocket(ScoringFunction):
    """
    🚄 — Calculates the distance between all atoms of the ligand and a pocket.

    This class uses a :class:`~pyrite.Pocket` object, which contains a number of alpha spheres describing
    the pocket. The distances to these alpha spheres are determined using a
    :class:`~scipy.spatial.KDTree` via a :class:`~pyrite.scoring.KNNDependency`.

    The score of this class is the sum of distances of all atoms of the ligand to the nearest
    alpha sphere in the pocket.

    .. warning::
        Currently, this class only supports :class:`~pyrite.Pocket` where the radius of each alpha
        sphere is equal. Using a :class:`~pyrite.Pocket` where this is not the case may result in
        unwanted results.


    **Speed**: 🚄–✈️, depending on the number of alpha spheres in the pocket.


    Parameters
    ----------
    molecule : Mol
        The molecule to be used for the calculation.
    pocket : Pocket
        The pocket to be used for the calculation.
    include_hs : bool, default False
        If `include_hs` is ``False``, Hydrogen atoms will be masked out of the calculation.


    """

    def __init__(
        self,
        molecule: Mol,
        pocket: Pocket,
        include_hs: bool = False,
    ):
        self.mol = molecule

        self.pocket = pocket
        self._include_hs = include_hs
        self._mask = np.array([include_hs or atom.GetAtomicNum() > 1 for atom in self.mol.atoms])

        self.cutoff = 40

        self._nn_dep = KNNDependency(
            pocket.centers,
            PositionQuery(self.mol),
            1,
            self.cutoff,
        )

    def get_dependencies(self) -> list[Dependency]:
        return [self._nn_dep]

    def _score(self, pose, computed) -> float:
        r, _, mask = computed[self._nn_dep]
        r, mask = r[..., 0], mask[..., 0]

        s = np.maximum(r - self.pocket.radii[0], 0.0)
        s[~mask] = self.cutoff
        s[~self._mask] = 0.0

        return float(np.sum(s))


class WeightedBoundsOverlap(ScoringFunction):
    """
    🚗 — Calculates the overlap between a ligand and a pocket. The overlap is weighted by the
    weights of the alpha spheres.

    This class uses a :class:`~pyrite.bounds.Pocket` object, which contains a number of alpha spheres
    describing the pocket. When the :class:`~pyrite.bounds.Pocket` is acquired from a ``pqr`` file,
    the charges are read as weights.

    The score of this class is the root of the mean of the square of the closest weights to each
    atom of the ligand.

    .. warning::
        Currently, this class only supports :class:`~pyrite.bounds.Pocket` where the radius of each
        alpha sphere is equal. Using :class:`~pyrite.bounds.Pocket` where this is not the case may
        result in unwanted results.


    **Speed**: 🚲–🚄, depending on the number of alpha spheres in the pocket.


    Parameters
    ----------
    molecule : Mol
        The molecule to be used for the calculation.
    pocket : Pocket
        The pocket to be used for the calculation.
    include_hs : bool, default False
        If `include_hs` is `False`, Hydrogen atoms will be masked out of the calculation.
    outside_penalty : float, optional
        This value is an optional multiplier of the maximum charge of the pocket. Any point outside
        the pocket is assigned this weight.

    """

    def __init__(
        self,
        molecule: Mol,
        pocket: Pocket,
        include_hs: bool = False,
        outside_penalty: float | None = None,
    ):
        self.mol = molecule

        self.pocket = pocket
        self.include_hs = include_hs
        self.mask = np.array([include_hs or atom.GetAtomicNum() > 1 for atom in self.mol.atoms])

        self.nn_dep = KNNDependency(
            pocket.centers,
            PositionQuery(self.mol, self.mask),
            1,
            4,
        )

        self.outside_penalty = outside_penalty if outside_penalty is not None else 0.0

        self.max_charge = np.max(self.pocket.charges) * self.outside_penalty

    def get_dependencies(self) -> list[Dependency]:
        return [self.nn_dep]

    def _score(self, pose, computed) -> float:
        r, idx, safe_mask = computed[self.nn_dep]
        r, idx, safe_mask = r[..., 0], idx[..., 0], safe_mask[..., 0]

        safe_idx = np.where(safe_mask, idx, 0)

        mask = r > self.pocket.radii[0]

        c = self.pocket.charges[safe_idx]
        c[mask | ~safe_mask] = self.max_charge

        # return float(np.sqrt(np.mean(c**2)))
        return float(np.sum(c))


class OutOfBoundsPenalty(ScoringFunction):
    """
    🐌 — Calculates the distance between all atoms of the ligand and the bounds.

    .. warning::
        This class is currently very slow, as the implementation is not vectorized, nor uses a
        :class:`~scipy.spatial.KDTree`.


    **Speed**: 🐌


    Parameters
    ----------
    molecule : Mol
        The molecule to be used for the calculation.
    bounds : Bounds
        The bounds to be used for the calculation. Using :class:`~pyrite.bounds.Pocket` is not supported,
        use :class:`~pyrite.scoring.DistanceToPocket` instead.

    Raises
    ------
    TypeError:
        If a :class:`~pyrite.bounds.Pocket` object is provided as bounds.


    See Also
    --------
    DistanceToPocket
        To calculate the distance to a :class:`~pyrite.bounds.Pocket`.

    """

    def __init__(self, molecule: Mol, bounds: Bounds):
        self.mol = molecule
        self.bounds = bounds
        self._position_dep = PositionDependency(molecule)

    def get_dependencies(self) -> list[Dependency]:
        return [self._position_dep]

    def _score(self, pose, computed) -> float:
        score = 0.0
        for position in computed[self._position_dep]:
            score += self.bounds.squared_distance(tuple(position))

        return score
