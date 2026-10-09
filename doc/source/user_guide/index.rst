User guide
==========

Pyrite is a Python toolkit for molecular docking: placing a ligand in the binding site of a
receptor, and finding the poses that fit best. It is built from separate pieces (molecules, poses,
scoring functions, binding sites and searches) that work together in a docking run, and just as
well on their own or with code of your own.

This guide explains those pieces, with code that runs. New to Pyrite? Start with
:doc:`getting_started`, a complete docking run on one page, and read on in order: every page builds
on the ones before it.

Start here
----------

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: Getting started
      :link: getting_started
      :link-type: doc

      A complete docking run, from loading the molecules to the docked poses, in a few steps.

The pieces
----------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Molecules
      :link: molecules
      :link-type: doc

      Loading ligands and receptors, hydrogens, atom types, rotatable bonds, and viewing them.

   .. grid-item-card:: Poses
      :link: poses
      :link-type: doc

      How a pose places a molecule: its rotation, translation and torsions, one pose or many.

   .. grid-item-card:: Scoring
      :link: scoring
      :link-type: doc

      Scoring functions: the terms, combining them, gradients, grids, and comparing poses.

   .. grid-item-card:: Searching
      :link: searching
      :link-type: doc

      Binding sites, starting poses, basin hopping, and choosing the best poses.

Going further
-------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Writing a scoring function
      :link: writing_scoring_functions
      :link-type: doc

      A new term in one method, and how to make it batched, fast, and fit for a grid.

   .. grid-item-card:: Making it fast
      :link: making_it_fast
      :link-type: doc

      Where the time goes, and what to do about it: grids, gradients, batches, and parallel runs.

Throughout the guide
--------------------

- The examples dock the inhibitor of the 2BOH crystal structure into factor Xa, so that the results
  can be checked against the crystal pose. Some use small molecules made from SMILES.
- Every page is run when the documentation is built: the outputs shown are real.
- Distances are in Angstrom, angles in radians, and a lower score is better.
- Every function and class in the guide links to its page in the :doc:`API reference
  </api/index>`.

.. toctree::
   :hidden:
   :maxdepth: 1
   :titlesonly:

   getting_started
   molecules
   poses
   scoring
   searching
   writing_scoring_functions
   making_it_fast
