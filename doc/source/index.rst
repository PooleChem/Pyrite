Pyrite
======

Pyrite is a Python toolkit for molecular docking: placing a ligand in the binding site of a
receptor, and finding the poses that fit best. It is built from separate pieces that work together
in a docking run, and just as well on their own or with code of your own:

- **Molecules and poses**: ligands and receptors from SDF, PDB or SMILES, and poses that place a
  molecule with a rotation, a translation and its torsions, one pose or a batch.
- **Scoring functions**: terms that are combined with ordinary arithmetic, scored one pose or many
  at a time, with gradients, and on precomputed grids for speed.
- **Searches**: binding sites and pockets, starting poses, basin hopping, and clustering of the
  results, or any optimizer of your own.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Getting started
      :link: user_guide/getting_started
      :link-type: doc

      A complete docking run, from loading the molecules to the docked poses, on one page.

   .. grid-item-card:: User guide
      :link: user_guide/index
      :link-type: doc

      The pieces of Pyrite, one page each, with code that runs.

   .. grid-item-card:: API reference
      :link: api/index
      :link-type: doc

      Every module, class and function, with its parameters and examples.

   .. grid-item-card:: GitHub
      :link: https://github.com/PooleChem/Pyrite

      The source code, issues, and the place to contribute.

Installing
----------

Pyrite is on PyPI as ``pyrite-chem``, and needs Python 3.10 or newer:

.. code-block:: bash

   pip install pyrite-chem
   pip install "pyrite-chem[pdbfixer]"  # also OpenMM and PDBFixer, to prepare receptor PDB files

The import name is ``pyrite``. Its dependencies (numpy, scipy, numba, RDKit, py3Dmol) come with it.
With conda, they can come from conda-forge instead:

.. code-block:: bash

   git clone https://github.com/PooleChem/Pyrite.git
   cd Pyrite
   conda env create -f doc/environment.yml
   conda activate pyrite-docs
   pip install -e .

On the newest macOS with Python 3.10, SciPy's pip wheels fail to load; use Python 3.11 or newer, or
SciPy from conda-forge.

.. toctree::
   :maxdepth: 1
   :hidden:
   :titlesonly:

   User Guide <user_guide/index>
   API Reference <api/index>
