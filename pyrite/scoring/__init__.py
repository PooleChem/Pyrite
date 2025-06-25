"""
=========================================
Scoring Functions (:mod:`pyrite.scoring`)
=========================================

.. currentmodule:: pyrite.scoring

The ``pyrite.scoring`` namespace holds all objects related to the scoring of ligand poses, as well
as utilities that can be used to manipulate, create, and implement new scoring functions.

.. autosummary::
   :toctree: generated/

   ScoringFunction
   ConstantTerm
   Clamp

.. _ligand_scoring_functions:

Ligand Scoring Functions
------------------------

This class of functions scores based on just the :class:`~pyrite.Ligand` pose.

.. autosummary::
   :toctree: generated/

   InternalOverlap
   InternalEnergy


.. _bounds_scoring_functions:

Bounds Scoring Functions
------------------------

These functions are based on the position of the atoms in the :class:`~pyrite.Ligand`
relative to indicated :class:`~pyrite.bounds.Bounds`.

.. autosummary::
   :toctree: generated/

   OutOfBoundsPenalty
   DistanceToPocket
   WeightedBoundsOverlap


.. _protein_scoring_functions:

Protein based Scoring Functions
-------------------------------

These functions are based on the position of the atoms in the :class:`~pyrite.Receptor` relative
to the :class:`~pyrite.Ligand`. This class of functions can lead to biochemically more accurate
representations, but in turn require more computational power.

.. autosummary::
   :toctree: generated/

   LJ
   PlantsPLP


Abstract Functions
++++++++++++++++++

These functions implement commonly used distance functions, that can be used to create new
scoring functions. They are abstract, and thus can/should not be used without subclassing.

.. autosummary::
   :toctree: generated/

   ~protein._KNNScoringFunction
   ~protein._SlopeStep
   ~protein._PLP
   ~protein._ChargeScoringFunction


Vina based Scoring Functions
++++++++++++++++++++++++++++

These functions are based on the vina scoring function, specifically, the
`gnina <https://github.com/gnina/gnina>`_ implementation.


.. autosummary::
   :toctree: generated/

   Gaussian
   Repulsion
   Hydrophobic
   NonHydrophobic
   NonDirHBond
   VDW
   NonDirHBondLJ
   ElectroStatic
   AD4Solvation



Constants
---------

.. autosummary::
   :toctree: generated/

   NumTors
   NumAtoms


.. _misc_scoring_functions:

Miscellaneous
-------------

.. autosummary::
   :toctree: generated/

   RMSD
   Crowding
   NumProteinAtomsWithinA


Dependencies
------------

The :mod:`~pyrite.scoring.dependencies` module holds all objects related to
:class:`~pyrite.scoring.dependencies.Dependency`, which
is used to limit the number of expensive operations performed during scoring.


"""

from ._base import *
from .dependencies import *

# from .openmm import *
from .protein import *
from .bounds import *
from .internal import *
from .misc import *
