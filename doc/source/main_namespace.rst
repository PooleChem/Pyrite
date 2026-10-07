==========================================
The main pyrite namespace (:mod:`pyrite`)
==========================================

.. module:: pyrite

.. currentmodule:: pyrite

The main ``pyrite`` namespace holds common objects such as `Mol`, as well as
`AtomType` and other constants.


.. autosummary::
   :toctree: generated/

   Mol


Furthermore, `Viewer` allows for simple visualization of protein structures, molecules, bounds, and poses, in
ipython notebooks.

.. autosummary::
   :toctree: generated/

   Viewer


Poses
-----

A pose places a :class:`Mol`: its rotation, translation and torsions, as one vector.

.. autosummary::
   :toctree: generated/

   Pose
   Poses
   PoseLayout


Atom constants
--------------

.. autosummary::
   :toctree: generated/

   AtomType
   vina_atom_consts


Submodules
----------

.. toctree::
    :hidden:

    bounds
    scoring
    dependencies
    search
    cluster
    io

===================================  ==================================================
:mod:`~pyrite.bounds`                Bounds functionality
:mod:`~pyrite.scoring`               Scoring functions
:mod:`~pyrite.scoring.dependencies`  Shared computations of scoring functions
:mod:`~pyrite.search`                Search for good poses
:mod:`~pyrite.cluster`               Clustering of poses
:mod:`~pyrite.io`                    Reading and preparing input files
===================================  ==================================================
