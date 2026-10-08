=========================================
Molecules and poses (:mod:`pyrite`)
=========================================

.. module:: pyrite

.. currentmodule:: pyrite

The main ``pyrite`` namespace holds the molecules that are docked, the poses that place them, and a
viewer for both.

Molecules
---------

A `Mol` is a ligand or a receptor: an RDKit molecule with the atom types and rotatable bonds that
docking needs.

.. autosummary::
   :toctree: generated/

   Mol
   AtomType
   vina_atom_consts

Poses
-----

A pose places a `Mol`: its rotation, translation and torsions, as one vector.

.. autosummary::
   :toctree: generated/

   Pose
   Poses
   PoseLayout

Viewing
-------

`Viewer` shows molecules, bounds and poses in 3D, in a notebook or a documentation page.

.. autosummary::
   :toctree: generated/

   Viewer
