---
file_format: mystnb
kernelspec:
  name: python3
---

# Searching

A docking search looks for the poses with the lowest score inside a binding site. This page goes
through **one example workflow**, the one of [Getting started](getting_started.md), step by step. It
is not the only way to use Pyrite: every step is a separate piece, which can be changed, replaced by
your own code, combined differently, or left out. The scoring functions work with any optimizer, and
the bounds, placements and clustering with any search. [Other searches](#other-searches), at the end,
shows an example.

The example workflow has four steps:

1. [The binding site](#the-binding-site): where the ligand may go.
2. [Starting poses](#starting-poses): many placements in the binding site, filtered to a few
   good and different ones.
3. [Basin hopping](#basin-hopping): a search from every starting pose.
4. [Choosing the poses](#choosing-the-poses): the best pose of every group of similar ones.

```{code-cell} python
import numpy as np

from pyrite import Mol, Poses, Viewer
from pyrite.io import fix_receptor_pdb

receptor = Mol.from_pdb(fix_receptor_pdb("input/2boh.pdb"), hydrogens="add")
ligand = Mol.from_sdf("input/2boh_ligand.sdf", flexible=True, hydrogens="add")
rng = np.random.default_rng(12)
```

## The binding site

The binding site is a {class}`~pyrite.bounds.Bounds`: a shape the ligand is searched within. The
simplest is a box. When the binding site is known from a ligand, as here, a box around that ligand,
with some room to spare, is made with {meth}`~pyrite.bounds.RectangularBounds.autobox`:

```{code-cell} python
from pyrite.bounds import RectangularBounds

box = RectangularBounds.autobox(ligand, padding=1.0)
```

Without a ligand, a box is placed by its size and center: `RectangularBounds((20, 20, 20),
at=(x, y, z))`. {class}`~pyrite.bounds.SphericalBounds` and
{class}`~pyrite.bounds.CylindricalBounds` are the other shapes.

The box bounds the *center atom* of the ligand: the rest of the ligand may stick out of it, as long
as it fits in the receptor.

A box also holds space the ligand cannot use, inside the receptor. A
{class}`~pyrite.bounds.Pocket` is the empty space itself: spheres that fill the cavities of the
receptor. {meth}`~pyrite.bounds.Pocket.from_mol` finds them, and
{meth}`~pyrite.bounds.Pocket.intersect` keeps those near the box:

```{code-cell} python
from pyrite.bounds import Pocket

pocket = Pocket.from_mol(receptor).intersect(box, padding=2.0)
len(pocket.centers)
```

```{code-cell} python
viewer = Viewer(width=600, height=400)
viewer.add(receptor, options={"surfacetype": None})
viewer.add(pocket, options={"wireframe": False, "opacity": 0.15})
viewer.add(ligand, box).zoom_to(ligand)
```

## Starting poses

A search starts from poses that are already in the binding site, with a sensible shape.
{func}`~pyrite.search.place_in` makes many: random positions in the pocket, each with a random
rotation and the torsions of one of a set of RDKit conformers:

```{code-cell} python
from pyrite.search import place_in

placements = place_in(
    ligand, pocket, n_positions=2000, n_conformations=20, rng=rng
)
len(placements)
```

Most of them clash with the receptor or stick out of the pocket. A cheap score of how well they fit
the pocket picks out the better ones, and {func}`~pyrite.search.boltzmann_diversity_filter` keeps a
set that fits well *and* is spread over the pocket, so that the searches do not all start in the
same place:

```{code-cell} python
from pyrite.scoring import DistanceToPocket, InternalOverlap
from pyrite.search import boltzmann_diversity_filter

fit = DistanceToPocket(ligand, pocket) + InternalOverlap(ligand)
energies = fit.batch_scores(placements)
starts = boltzmann_diversity_filter(placements, energies, k=32, rng=rng)
len(starts)
```

```{code-cell} python
viewer = Viewer(width=600, height=400)
viewer.add(receptor, options={"surfacetype": None})
viewer.add_v(ligand, starts, slider=False)
```

## Basin hopping

{class}`~pyrite.search.BasinHopping` searches from one starting pose. It repeats two steps: a
random *hop* to a nearby pose, followed by a local minimization to the bottom of the basin it lands
in. The new minimum is accepted if it is better, and sometimes when it is worse, so that the search
can climb out of a basin; the temperature `T`, in units of the score, decides how often.

The scoring function is {func}`~pyrite.scoring.vina_like_grid`: on a grid, its gradient is fast,
which the local minimization uses (`"jac": True`). The bounds of the box keep the minimization
inside it.

```{code-cell} python
from pyrite.scoring import vina_like_grid
from pyrite.search import BasinHopping, adaptive_stepsize, random_hop

score = vina_like_grid(ligand, receptor, box)
hopping = BasinHopping(
    score.get_score_and_gradient,
    random_hop(box.get_translation_bounds()),
    T=1.0,
    stepsize=0.5,
    adapt_stepsize=adaptive_stepsize(),
    minimizer_kwargs={
        "method": "L-BFGS-B",
        "jac": True,
        "bounds": box.get_bounds(ligand.layout),
    },
    rng=rng,
)
```

- `hop`: {func}`~pyrite.search.random_hop` moves, rotates and turns the torsions of a pose, by an
  amount set by the step size. It is a function `hop(pose, rng, stepsize)`, and can be replaced.
- `stepsize` and `adapt_stepsize`: {func}`~pyrite.search.adaptive_stepsize` makes the hops larger
  when most are accepted, and smaller when few are.
- `T` can also change during the run, with a schedule such as
  {func}`~pyrite.search.geometric_annealing`.

{meth}`~pyrite.search.BasinHopping.run` searches from one pose, for a number of hops. The result
holds the best pose and its score, and what happened in every hop:

```{code-cell} python
result = hopping.run(starts[0], niter=50)
result.fun, float(result.accepted.mean())
```

```{code-cell} python
import matplotlib.pyplot as plt

hops = np.arange(1, len(result.scores) + 1)
plt.figure(figsize=(6, 3))
accepted = result.accepted
plt.plot(hops, result.scores, ".", color="grey", label="minimum after the hop")
plt.plot(hops[accepted], result.scores[accepted], ".", label="accepted")
plt.plot(hops, np.minimum.accumulate(result.scores), label="best so far")
plt.xlabel("Hop")
plt.ylabel("Score")
plt.legend()
plt.show()
```

## Many starts

One search can get stuck in one region of the pocket; searches from different starting poses
cover more. Every run is independent, so they can also run in parallel:

```{code-cell} python
results = [hopping.run(pose, niter=50) for pose in starts]
poses = Poses.from_list([result.x for result in results])
scores = np.array([result.fun for result in results])
```

## Choosing the poses

Many searches end in the same pose. {func}`~pyrite.cluster.cluster_and_select` groups poses that
are within an RMSD `cutoff` of each other, and keeps the best of every group, best first:

```{code-cell} python
from pyrite.cluster import cluster_and_select

best = cluster_and_select(ligand, poses, scores, cutoff=2.0, n_output=5)
scores[best]
```

{func}`~pyrite.cluster.rmsd_matrix` gives the RMSD between every pair of poses, for clustering of
your own. Here the crystal pose is known, so the poses can be compared with it:

```{code-cell} python
from pyrite.scoring import RMSD

rmsd = RMSD(ligand)
[round(rmsd.get_score(poses[i]), 2) for i in best]
```

```{code-cell} python
crystal = ligand.copy()
crystal.set_draw_options({"colorscheme": "greenCarbon"})
viewer = Viewer(width=600, height=400)
viewer.add(receptor, options={"surfaceopacity": 0.4})
viewer.add(crystal).add_v(ligand, poses[best])
```

The search is random: with another seed, or fewer starting poses or hops, it does not always find
the best pose. More of both make that more likely, at the cost of time; see Making it fast.

## Other searches

Nothing above is fixed. Some of the ways to do it differently:

- Another **scoring function** in the same search: any scoring function, combined as on
  {doc}`scoring`, or one of your own ({doc}`writing_scoring_functions`).
- Another **hop** or **temperature schedule** for basin hopping: both are plain functions.
- Only a **local minimization** of the starting poses, with {func}`scipy.optimize.minimize`.
- **Rescoring** the poses of one search with another scoring function, such as an exact one, or
  one with {class}`~pyrite.scoring.InternalEnergy`.
- Another **optimizer** altogether. A scoring function is a function of a pose, and a pose is a
  numpy array, so any optimizer works with it.

For example, {func}`scipy.optimize.differential_evolution` evolves a population of poses within the
bounds of the box. It can score the whole population at once, with
{meth}`~pyrite.scoring.ScoringFunction.batch_scores`:

```{code-cell} python
from scipy.optimize import differential_evolution

from pyrite import Pose

bounds = box.get_bounds(ligand.layout, rotation_bounds=(-np.pi, np.pi))
evolved = differential_evolution(
    lambda population: score.batch_scores(population.T),
    bounds,
    vectorized=True,
    updating="deferred",
    seed=0,
)
pose = Pose(evolved.x, ligand.layout)
round(float(evolved.fun), 2), round(rmsd.get_score(pose), 2)
```

Here it is fast, but finds a worse pose than basin hopping did: which search works best depends on
the problem.
