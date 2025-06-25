==========================================
The main pyrite namespace (:mod:`pyrite`)
==========================================

.. module:: pyrite

.. currentmodule:: pyrite

The main ``pyrite`` namespace holds common objects such as `Ligand` and `Receptor`, as well as
`AtomTypes` and other constants.


.. autosummary::
   :toctree: generated/
   :template: autoclass_ligand_no_rdkit.rst

   Ligand
   Receptor


Furthermore, `Viewer` allows for simple visualization of protein structures, molecules, bounds, and poses, in
ipython notebooks.

.. autosummary::
   :toctree: generated/

   Viewer


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

======================  ==================================================
:mod:`~pyrite.bounds`   Bounds functionality
:mod:`~pyrite.scoring`  Scoring functions
======================  ==================================================
