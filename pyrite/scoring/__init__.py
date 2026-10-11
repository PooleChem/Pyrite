"""
=========================================
Scoring functions (:mod:`pyrite.scoring`)
=========================================

.. currentmodule:: pyrite.scoring

The ``pyrite.scoring`` namespace holds all objects related to the scoring of poses, as well as
utilities that can be used to manipulate, create, and implement new scoring functions.

The documentation of every scoring function grades its speed: the time to score one pose, in
orders of magnitude.

====  =================
🚀    under 3 µs
✈️    3 µs to 30 µs
🚗    30 µs to 300 µs
🚲    0.3 ms to 3 ms
🐢    3 ms to 30 ms
🐌    over 30 ms
====  =================

The grades were measured on a ligand of 33 heavy atoms in a protein, with the default parameters
(see ``benchmarks/speed_grades.py`` in the repository). They are a guide, not a promise: they
depend on the computer, the molecules and the parameters. "batched" is the time per pose in a
batch (:meth:`ScoringFunction.batch_scores`), given where it is better by a grade.

Ready-made scoring functions
----------------------------

.. autosummary::
   :toctree: generated/

   vina_like
   vina_like_grid

Combining scoring functions
---------------------------

Every scoring function is a :class:`ScoringFunction`. They are combined with ``+``, ``-``,
``*``, ``/`` and ``**`` into one scoring function, which shares the work of its terms.

.. autosummary::
   :toctree: generated/

   ScoringFunction
   Clamp
   ConstantTerm

.. _protein_scoring_functions:

Ligand-receptor scoring functions
---------------------------------

These functions score the atoms of one :class:`~pyrite.Mol` (the ligand, whose pose changes)
against their nearest neighbors in a fixed :class:`~pyrite.Mol` (the receptor). The terms on the
same receptor share one nearest neighbor search.

Vina-like terms, after the `gnina <https://github.com/gnina/gnina>`_ implementation of Vina:

.. autosummary::
   :toctree: generated/

   Gaussian
   Repulsion
   Hydrophobic
   NonHydrophobic
   NonDirHBond

Lennard-Jones terms:

.. autosummary::
   :toctree: generated/

   LJ
   VDW
   NonDirHBondLJ

Terms based on Gasteiger charges:

.. autosummary::
   :toctree: generated/

   ElectroStatic
   AD4Solvation

The PLANTS piecewise linear potential, and a count of receptor contacts:

.. autosummary::
   :toctree: generated/

   PlantsPLP
   NumProteinAtomsWithinA

Grid scoring
------------

A KNN-based scoring function can be computed once on a grid around the binding site, and
interpolated during a search: much faster, with an analytic gradient. Import it from
``pyrite.scoring.grid``.

.. autosummary::
   :toctree: generated/

   ~grid.GridScore

.. _ligand_scoring_functions:

Ligand-only scoring functions
-----------------------------

These functions only depend on the pose of the ligand itself: its internal clashes and energy, or
a count that is the same for every pose.

.. autosummary::
   :toctree: generated/

   InternalOverlap
   InternalEnergy
   NumTors
   NumAtoms

.. _bounds_scoring_functions:

Binding site scoring functions
------------------------------

These functions keep the ligand in a binding site: they depend on the position of its atoms
relative to :class:`~pyrite.bounds.Bounds` or a :class:`~pyrite.bounds.Pocket`.

.. autosummary::
   :toctree: generated/

   DistanceToPocket
   WeightedBoundsOverlap
   OutOfBoundsPenalty

.. _misc_scoring_functions:

Comparing poses
---------------

.. autosummary::
   :toctree: generated/

   RMSD
   Crowding

Building blocks for new scoring functions
-----------------------------------------

These abstract classes implement what a family of scoring functions shares, so that a new one
only implements what is different: a kernel of the distance, which atom pairs count, or a score
from an RDKit molecule. They cannot be used without subclassing. How to write a new scoring
function, with these or from scratch, is explained in :doc:`/user_guide/writing_scoring_functions`.

.. autosummary::
   :toctree: generated/

   ~protein._KNNScoringFunction
   ~protein._SlopeStep
   ~protein._ChargeScoringFunction
   ~protein._PLP
   ~_base._RDKitScoringFunction

Dependencies
------------

The :mod:`~pyrite.scoring.dependencies` module holds all objects related to
:class:`~pyrite.scoring.dependencies.Dependency`, which is used to share expensive computations
between scoring functions, so that they are only done once.

"""

from ._base import (
    Clamp,
    ConstantTerm,
    ScoringFunction,
)
from .bounds import (
    DistanceToPocket,
    OutOfBoundsPenalty,
    WeightedBoundsOverlap,
)
from .dependencies import (
    Dependency,
    KDTreeCache,
    KNNDependency,
)
from .internal import (
    InternalEnergy,
    InternalOverlap,
)
from .misc import (
    RMSD,
    Crowding,
    NumAtoms,
    NumProteinAtomsWithinA,
    NumTors,
)
from .presets import vina_like, vina_like_grid
from .protein import (
    LJ,
    VDW,
    AD4Solvation,
    ElectroStatic,
    Gaussian,
    Hydrophobic,
    NonDirHBond,
    NonDirHBondLJ,
    NonHydrophobic,
    PlantsPLP,
    Repulsion,
)

__all__ = [
    "AD4Solvation",
    "Clamp",
    "ConstantTerm",
    "Crowding",
    "Dependency",
    "DistanceToPocket",
    "ElectroStatic",
    "Gaussian",
    "Hydrophobic",
    "InternalEnergy",
    "InternalOverlap",
    "KDTreeCache",
    "KNNDependency",
    "LJ",
    "NonDirHBond",
    "NonDirHBondLJ",
    "NonHydrophobic",
    "NumAtoms",
    "NumProteinAtomsWithinA",
    "NumTors",
    "OutOfBoundsPenalty",
    "PlantsPLP",
    "RMSD",
    "Repulsion",
    "ScoringFunction",
    "VDW",
    "WeightedBoundsOverlap",
    "vina_like",
    "vina_like_grid",
]
