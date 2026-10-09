---
file_format: mystnb
kernelspec:
  name: python3
---

# Getting started

This page docks a ligand into its receptor, start to finish: load the molecules, build a scoring
function, search for the best poses, and look at the result. Every step is explained in more detail
on the pages that follow.

This is one example workflow, not the only one: Pyrite is a set of pieces (molecules, poses,
scoring functions, bounds, searches) that can be combined in other ways, or with code of your own.
See [Searching](searching.md) for some of the alternatives.

The example is 2BOH, the blood-clotting enzyme factor Xa with an inhibitor bound. The crystal
structure tells us where the ligand really binds, so we can check the result.

## Installing

Pyrite is not a package on PyPI yet. Clone the repository, and put its folder on the Python path:

```bash
git clone https://github.com/PooleChem/Pyrite.git
export PYTHONPATH="$PWD/Pyrite:$PYTHONPATH"
```

## Loading the molecules

A receptor is loaded from a PDB file, and a ligand from an SDF file (or a SMILES string). A PDB file
straight from the Protein Data Bank also holds waters, ions, the ligand itself and sometimes
non-standard residues: {func}`~pyrite.io.fix_receptor_pdb` leaves only the protein. It has no
hydrogens either, and the hydrogens decide which atoms can donate a hydrogen bond, so they are added
when it is loaded.

```{code-cell} python
import numpy as np

from pyrite import Mol, Poses, Viewer
from pyrite.io import fix_receptor_pdb

receptor = Mol.from_pdb(fix_receptor_pdb("input/2boh.pdb"), hydrogens="add")
ligand = Mol.from_sdf("input/2boh_ligand.sdf", flexible=True)
ligand.n_tors
```

The ligand is `flexible`: a search may turn its {attr}`~pyrite.Mol.n_tors` rotatable bonds. The pose
it was loaded in, the crystal pose, is kept as {attr}`~pyrite.Mol.input_pose`.

```{code-cell} python
Viewer(receptor, ligand, width=600, height=400).zoom_to(ligand)
```

## The binding site

The search looks for poses in a box around the binding site. Here we know where the ligand binds,
so the box is put around the crystal pose, with some room to spare:

```{code-cell} python
from pyrite.bounds import RectangularBounds

box = RectangularBounds.autobox(ligand, padding=1.0)
```

## The scoring function

A scoring function gives every pose a score: lower is better.
{func}`~pyrite.scoring.vina_like_grid` is close to the scoring function of AutoDock Vina: two attractive
Gaussians, a repulsion, hydrophobic contacts and hydrogen bonds between the ligand and the receptor,
a term for the ligand clashing with itself, all divided by a penalty for the number of rotatable
bonds.

```{code-cell} python
from pyrite.scoring import vina_like_grid

score = vina_like_grid(ligand, receptor, box)
round(score.get_score(ligand.input_pose), 2)  # the crystal pose
```

The receptor terms are put on a grid over the box: they are computed once, after which
scoring a pose, and its gradient, is fast. A scoring function is built from terms with ordinary
arithmetic, and `vina_like_grid` is a shorthand for one; the Scoring page builds it, and others, term by
term.

## Starting poses

The search starts from a set of different poses inside the binding pocket.
{func}`~pyrite.search.place_in` places many random conformations in the pocket, and
{func}`~pyrite.search.boltzmann_diversity_filter` keeps a diverse set of the ones that fit best.

```{code-cell} python
from pyrite.bounds import Pocket
from pyrite.scoring import DistanceToPocket, InternalOverlap
from pyrite.search import boltzmann_diversity_filter, place_in

rng = np.random.default_rng(5)
pocket = Pocket.from_mol(receptor).intersect(box, padding=2.0)
placements = place_in(ligand, pocket, n_positions=2000, n_conformations=20, rng=rng)
fit = DistanceToPocket(ligand, pocket) + InternalOverlap(ligand)
starts = boltzmann_diversity_filter(placements, fit.batch_scores(placements), k=32, rng=rng)
```

## Searching

{class}`~pyrite.search.BasinHopping` alternates random jumps with local minimizations, from every
starting pose. The minimizer uses the gradient of the scoring function, and stays within the box.

```{code-cell} python
from pyrite.search import BasinHopping, adaptive_stepsize, random_hop

hopping = BasinHopping(
    score.get_score_and_gradient,
    random_hop(box.get_translation_bounds()),
    T=1.0,
    stepsize=0.5,
    adapt_stepsize=adaptive_stepsize(),
    minimizer_kwargs={"method": "L-BFGS-B", "jac": True, "bounds": box.get_bounds(ligand.layout)},
    rng=rng,
)
results = [hopping.run(pose, niter=50) for pose in starts]
poses = Poses.from_list([result.x for result in results])
scores = np.array([result.fun for result in results])
```

## The results

Many runs end in the same pose. {func}`~pyrite.cluster.cluster_and_select` groups poses that are
within 2 Angstrom RMSD of each other, and keeps the best of every group:

```{code-cell} python
from pyrite.cluster import cluster_and_select
from pyrite.scoring import RMSD

best = cluster_and_select(ligand, poses, scores, cutoff=2.0, n_output=5)
rmsd = RMSD(ligand)  # to the crystal pose
for i in best:
    print(f"score {scores[i]:6.2f}   RMSD to the crystal pose {rmsd.get_score(poses[i]):5.2f} A")
```

The best pose is within 1 Angstrom of the crystal pose. A search is random: with another seed, or
fewer starting poses or iterations, it does not always find that pose, and more of both make it more
likely.

The poses in the binding site, with the crystal pose in green; the slider steps through them:

```{code-cell} python
crystal = ligand.copy()
crystal.set_draw_options({"colorscheme": "greenCarbon"})
Viewer(receptor, crystal, width=600, height=400, options={"surfaceopacity": 0.4}).add_v(
    ligand, poses[best]
)
```

Finally, the poses are saved to an SDF file, best first:

```{code-cell} python
ligand.to_sdf("docked.sdf", poses=poses[best])
```

```{code-cell} python
:tags: [remove-cell]

import os

os.remove("docked.sdf")
```
