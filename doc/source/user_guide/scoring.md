---
file_format: mystnb
kernelspec:
  name: python3
---

# Scoring

A scoring function gives a pose a score: lower is better. Docking searches for the pose with the
lowest score, so the scoring function decides what a good pose is. In Pyrite, a scoring function is
built from terms, combined with ordinary arithmetic.

```{code-cell} python
import numpy as np

from pyrite import Mol, Pose, Poses
from pyrite.io import fix_receptor_pdb

receptor = Mol.from_pdb(fix_receptor_pdb("input/2boh.pdb"), hydrogens="add")
ligand = Mol.from_sdf("input/2boh_ligand.sdf", flexible=True, hydrogens="add")
pose = ligand.input_pose
```

## Terms

A term scores one aspect of a pose. {class}`~pyrite.scoring.Repulsion`, for example, scores the
ligand atoms that overlap with receptor atoms: in the crystal pose there is little overlap, and
pushing the ligand 2 Angstrom into the receptor makes much more.

```{code-cell} python
from pyrite.scoring import Repulsion

repulsion = Repulsion(ligand, receptor)

moved = Pose(np.asarray(pose).copy(), ligand.layout)
moved.translation += [2.0, 0.0, 0.0]
repulsion.get_score(pose), repulsion.get_score(moved)
```

The terms come in a few families; see {mod}`pyrite.scoring` for all of them:

- **Between ligand and receptor**: every ligand atom against its nearest receptor atoms. The terms
  of Vina ({class}`~pyrite.scoring.Gaussian`, {class}`~pyrite.scoring.Repulsion`,
  {class}`~pyrite.scoring.Hydrophobic`, {class}`~pyrite.scoring.NonDirHBond`), Lennard-Jones
  terms, terms based on charges, and the PLANTS potential.
- **The ligand alone**: its clashes with itself ({class}`~pyrite.scoring.InternalOverlap`), its
  internal energy ({class}`~pyrite.scoring.InternalEnergy`), and counts such as
  {class}`~pyrite.scoring.NumTors`.
- **A binding site**: how far the ligand is outside a box or pocket, such as
  {class}`~pyrite.scoring.DistanceToPocket`.
- **Other poses**: {class}`~pyrite.scoring.RMSD` to a reference pose.

The documentation of every term grades how fast it is, in orders of magnitude of the time to score
one pose: 🚀 (under 3 µs), ✈️, 🚗, 🚲, 🐢 and 🐌 (over 30 ms); see {mod}`pyrite.scoring`.

## Combining terms

Terms are combined with `+`, `-`, `*`, `/` and `**`, with numbers and with each other. The result is
a scoring function like any other. This one is close to the scoring function of AutoDock Vina:

```{code-cell} python
from pyrite.scoring import Gaussian, Hydrophobic, InternalOverlap, NonDirHBond, NumTors

gauss1 = Gaussian(ligand, receptor, offset=0.0, width=0.5)
gauss2 = Gaussian(ligand, receptor, offset=3.0, width=2.0)
hydrophobic = Hydrophobic(ligand, receptor, good=0.5, bad=1.5)
hbond = NonDirHBond(ligand, receptor, good=-0.7, bad=0.0)

receptor_terms = (
    -0.035579 * gauss1
    - 0.005156 * gauss2
    + 0.840245 * repulsion
    - 0.035069 * hydrophobic
    - 0.587439 * hbond
)
score = (receptor_terms + 0.5 * InternalOverlap(ligand)) / (1 + 0.05846 * NumTors(ligand))
score.get_score(pose)
```

{func}`~pyrite.scoring.vina_like` is a shorthand for exactly this scoring function:

```{code-cell} python
from pyrite.scoring import vina_like

vina_like(ligand, receptor).get_score(pose)
```

Terms that need the same work share it: the five receptor terms above look up the nearest receptor
atoms of every ligand atom once, not five times.

To see what every term contributes, pass a dictionary to {meth}`~pyrite.scoring.ScoringFunction.get_score`.
It is filled with the score of every part of the scoring function:

```{code-cell} python
parts = {}
score.get_score(pose, subscores=parts)
{name: round(parts[term], 3) for name, term in
 [("gauss1", gauss1), ("gauss2", gauss2), ("repulsion", repulsion),
  ("hydrophobic", hydrophobic), ("hbond", hbond)]}
```

A score can be limited with {meth}`~pyrite.scoring.ScoringFunction.clamp`, for example so that one
large clash does not dominate:

```{code-cell} python
repulsion.clamp(max_score=10.0).get_score(moved)
```

## Many poses at once

{meth}`~pyrite.scoring.ScoringFunction.batch_scores` scores a whole batch of poses at once. How much
faster that is than one pose at a time depends on the terms: cheap terms and grids are several times
faster in a batch, while terms between ligand and receptor are about as fast either way, as their
work is per pair of atoms. Here, the crystal pose shifted at random by about an Angstrom:

```{code-cell} python
rng = np.random.default_rng(0)
n = 50
poses = Poses.from_parts(
    np.tile(pose.rotation, (n, 1)),
    pose.translation + rng.normal(0.0, 1.0, (n, 3)),
    np.tile(pose.torsions, (n, 1)),
)
scores = score.batch_scores(poses)
float(scores.min()), float(np.median(scores))
```

## Gradients

A local optimizer needs the gradient of the score: how it changes with every variable of the pose.
{meth}`~pyrite.scoring.ScoringFunction.get_score_and_gradient` gives both:

```{code-cell} python
value, gradient = score.get_score_and_gradient(pose)
gradient.shape
```

Some terms know their gradient; for the others it is computed by finite differences, which costs
two extra scores for every variable of the pose. Grids, below, have their gradient for free.

## Grids

The terms between ligand and receptor are the expensive ones: every ligand atom is compared to the
receptor atoms around it. As the receptor does not move, they can be computed once, on a grid over
the binding site, and looked up during a search. {class}`~pyrite.scoring.grid.GridScore` puts terms
on a grid; {func}`~pyrite.scoring.vina_like_grid` is `vina_like` with its receptor terms on a grid:

```{code-cell} python
from pyrite.bounds import RectangularBounds
from pyrite.scoring import vina_like_grid

box = RectangularBounds.autobox(ligand, padding=1.0)
grid_score = vina_like_grid(ligand, receptor, box)
grid_score.get_score(pose), score.get_score(pose)
```

Building the grid takes a while; after that, scoring and its gradient are fast. The grid is
interpolated between its points, so its scores are close to, but not the same as, the exact ones:

```{code-cell} python
float(np.abs(grid_score.batch_scores(poses) - scores).max())
```

A finer `spacing` is closer to the exact scores, and more expensive to build. The grid only covers
the box: an atom outside it scores zero.

Only terms between ligand and receptor whose score depends on nothing but the type and position of
each ligand atom can be put on a grid: today, the Vina terms. Terms of the ligand alone, such as
`InternalOverlap` and `NumTors`, are combined with the grid, as `vina_like_grid` does. See
{doc}`writing_scoring_functions` for what a grid needs from a term.

## Comparing poses

{class}`~pyrite.scoring.RMSD` is the root mean square distance between the heavy atoms of a pose and
a reference pose, taking the symmetry of the molecule into account. By default, the reference is
the pose the molecule was loaded in:

```{code-cell} python
from pyrite.scoring import RMSD

RMSD(ligand).get_score(moved)
```

As a scoring function, an RMSD can also be part of a score, for example to keep a search close to a
known pose.

## Writing your own

A new term is a class with one method: see {doc}`writing_scoring_functions`.
