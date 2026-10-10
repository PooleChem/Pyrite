from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .internal import InternalEnergy, InternalOverlap
from .misc import NumTors
from .protein import Gaussian, Hydrophobic, NonDirHBond, Repulsion

if TYPE_CHECKING:
    from pyrite import Mol
    from pyrite.bounds import Bounds

    from ._base import ScoringFunction

# The weights of AutoDock Vina, as in gnina.
VINA_WEIGHTS = {
    "gauss1": -0.035579,
    "gauss2": -0.005156,
    "repulsion": 0.840245,
    "hydrophobic": -0.035069,
    "hydrogen_bond": -0.587439,
    "rotatable_bonds": 0.05846,
}


def _vina_receptor_terms(ligand: Mol, receptor: Mol) -> ScoringFunction:
    w = VINA_WEIGHTS
    return (
        w["gauss1"] * Gaussian(ligand, receptor, offset=0.0, width=0.5)
        + w["gauss2"] * Gaussian(ligand, receptor, offset=3.0, width=2.0)
        + w["repulsion"] * Repulsion(ligand, receptor)
        + w["hydrophobic"] * Hydrophobic(ligand, receptor, good=0.5, bad=1.5)
        + w["hydrogen_bond"] * NonDirHBond(ligand, receptor, good=-0.7, bad=0.0)
    )


def _vina_combine(ligand: Mol, receptor_terms: ScoringFunction, internal) -> ScoringFunction:
    if internal == "overlap":
        internal = 0.5 * InternalOverlap(ligand)
    elif internal == "energy":
        internal = 0.01 * InternalEnergy(ligand)
    elif isinstance(internal, str):
        raise ValueError(f"Unknown internal term {internal!r}: use 'overlap', 'energy' or None.")

    score = receptor_terms if internal is None else receptor_terms + internal
    return score / (1 + VINA_WEIGHTS["rotatable_bonds"] * NumTors(ligand))


def vina_like(
    ligand: Mol,
    receptor: Mol,
    *,
    internal: Literal["overlap", "energy"] | ScoringFunction | None = "overlap",
) -> ScoringFunction:
    """Return a scoring function like that of AutoDock Vina.

    The Vina terms between the ligand and the receptor, with the weights of Vina: two attractive
    Gaussians, a repulsion, hydrophobic contacts and hydrogen bonds, plus a term for the ligand
    clashing with itself, all divided by a penalty for the number of rotatable bonds::

        (receptor_terms + internal) / (1 + 0.05846 * NumTors(ligand))

    The terms are computed exactly, and the gradient by finite differences. For a search, the
    same scoring function on a grid, :func:`vina_like_grid`, is much faster.

    It is a shorthand: the same scoring function can be written out, and changed, term by term.

    .. note::
        Vina scores the ligand against itself with the same terms as against the receptor. Here
        the intramolecular term is :class:`~pyrite.scoring.InternalOverlap` or
        :class:`~pyrite.scoring.InternalEnergy` instead, so the scores are close to, but not the
        same as, those of Vina.

    Parameters
    ----------
    ligand : Mol
        The ligand, whose poses are scored.
    receptor : Mol
        The receptor.
    internal : {'overlap', 'energy'} or ScoringFunction or None, default 'overlap'
        The term for the ligand against itself: ``0.5 * InternalOverlap(ligand)`` (fast),
        ``0.01 * InternalEnergy(ligand)`` (the MMFF energy: slower, and best with all hydrogens
        on the ligand), a scoring function of your own, or None for no such term.

    Returns
    -------
    ScoringFunction
        The scoring function.

    Raises
    ------
    ValueError
        If `internal` is an unknown string.

    See Also
    --------
    vina_like_grid : The same, with the receptor terms on a grid.
    pyrite.scoring.Gaussian : One of the terms.

    Examples
    --------
    >>> score = vina_like(ligand, receptor)
    >>> score.get_score(ligand.input_pose)

    With the MMFF energy of the ligand as the intramolecular term:

    >>> score = vina_like(ligand, receptor, internal="energy")
    """
    return _vina_combine(ligand, _vina_receptor_terms(ligand, receptor), internal)


def vina_like_grid(
    ligand: Mol,
    receptor: Mol,
    binding_site: Bounds,
    *,
    spacing: float = 0.5,
    internal: Literal["overlap", "energy"] | ScoringFunction | None = "overlap",
) -> ScoringFunction:
    """Return a scoring function like that of AutoDock Vina, with the receptor terms on a grid.

    The same scoring function as :func:`vina_like`, with its terms between the ligand and the
    receptor put on a :class:`~pyrite.scoring.grid.GridScore` over the binding site. Building the
    grid takes a while; after that, scoring a pose and its (analytic) gradient is fast. The
    intramolecular term and the penalty for rotatable bonds are not on the grid.

    Parameters
    ----------
    ligand : Mol
        The ligand, whose poses are scored.
    receptor : Mol
        The receptor.
    binding_site : Bounds
        The region the ligand is searched within: the grid covers it.
    spacing : float, default 0.5
        The spacing of the grid, in Angstrom.
    internal : {'overlap', 'energy'} or ScoringFunction or None, default 'overlap'
        The term for the ligand against itself, see :func:`vina_like`.

    Returns
    -------
    ScoringFunction
        The scoring function.

    Raises
    ------
    ValueError
        If `internal` is an unknown string.

    See Also
    --------
    vina_like : The same, computed exactly.
    pyrite.scoring.grid.GridScore : Puts terms on a grid.

    Examples
    --------
    >>> from pyrite.bounds import RectangularBounds
    >>> box = RectangularBounds.autobox(ligand, padding=1.0)
    >>> score = vina_like_grid(ligand, receptor, box)
    >>> score, gradient = score.get_score_and_gradient(ligand.input_pose)
    """
    from .grid import GridScore

    grid = GridScore(_vina_receptor_terms(ligand, receptor), binding_site, spacing=spacing)
    return _vina_combine(ligand, grid, internal)
