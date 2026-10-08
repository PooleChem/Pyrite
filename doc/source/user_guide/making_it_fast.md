---
file_format: mystnb
kernelspec:
  name: python3
---

# Making it fast

Nearly all the time of a docking run goes into scoring poses: a search scores thousands of poses
per start. This page is about making that cheaper. The timings on it are measured when the page is
built, so they are those of one computer; what matters is how they compare.

```{code-cell} python
import time

import numpy as np

from pyrite import Mol, Poses
from pyrite.bounds import RectangularBounds
from pyrite.io import fix_receptor_pdb

receptor = Mol.from_pdb(fix_receptor_pdb("input/2boh.pdb"), hydrogens="add")
ligand = Mol.from_sdf("input/2boh_ligand.sdf", flexible=True, hydrogens="add")
box = RectangularBounds.autobox(ligand, padding=1.0)
pose = ligand.input_pose
```

## Measuring

Measure before changing anything. Two things make a timing wrong:

- **The first call** of many scoring functions compiles code (with numba) or builds caches: time
  only after a first call.
- **Scoring the same pose over and over** is faster than scoring different poses, as the
  computer's caches keep what the last call used: time on different poses, as a search scores
  them.

A small helper that does both, on poses spread around the crystal pose:

```{code-cell} python
rng = np.random.default_rng(0)
n = 256
poses = Poses.from_parts(
    pose.rotation + rng.normal(0.0, 0.2, (n, 3)),
    pose.translation + rng.normal(0.0, 0.5, (n, 3)),
    pose.torsions + rng.normal(0.0, 0.3, (n, ligand.n_tors)),
)


def microseconds_per_pose(scoring_function, batched=False):
    """The time to score one pose, in microseconds, over different poses."""
    def score_all():
        if batched:
            scoring_function.batch_scores(poses)
        else:
            for p in poses:
                scoring_function.get_score(p)

    score_all()  # the first call compiles and caches
    start = time.perf_counter()
    score_all()
    return round((time.perf_counter() - start) / n * 1e6, 1)
```

The documentation of every scoring function also grades its speed, from 🚀 to 🐌 (see
{mod}`pyrite.scoring`); `benchmarks/speed_grades.py` in the repository measures them all.

## Grids

The terms between ligand and receptor are by far the most expensive part of a scoring function.
As the receptor does not move, they can be computed once on a grid over the binding site, and
looked up during the search, with {class}`~pyrite.scoring.grid.GridScore`, or
{func}`~pyrite.scoring.vina_like_grid` for the Vina-like scoring function:

```{code-cell} python
from pyrite.scoring import vina_like, vina_like_grid

exact = vina_like(ligand, receptor)
start = time.perf_counter()
grid = vina_like_grid(ligand, receptor, box)
build_seconds = round(time.perf_counter() - start, 1)

microseconds_per_pose(exact), microseconds_per_pose(grid), build_seconds
```

Scoring on the grid is many times faster; building it takes seconds, so a grid pays off once a
search scores more poses than that. For a docking run, which scores hundreds of thousands, it
always does. The costs:

- **Building**: the grid scores every grid point. Its size grows with the cube of `1 / spacing`
  and of the size of the binding site: halving the spacing makes it 8 times as expensive. Build it
  once, and use it for every search in the same binding site.
- **Accuracy**: the grid is interpolated between its points (tricubic by default), so its scores
  are close to, not the same as, the exact ones. A finer spacing is closer.
- **Only the binding site**: an atom outside the grid scores zero.

## Gradients

The local minimization in a search needs the gradient of the score. A scoring function that knows
its gradient computes it with the score, at little extra cost. For the others, it is computed by
finite differences: two extra scores for every variable of the pose, here 24 for a ligand with 6
torsions.

```{code-cell} python
def microseconds_per_gradient(scoring_function):
    scoring_function.get_score_and_gradient(pose)
    start = time.perf_counter()
    for p in poses[:32]:
        scoring_function.get_score_and_gradient(p)
    return round((time.perf_counter() - start) / 32 * 1e6, 1)


microseconds_per_gradient(exact), microseconds_per_gradient(grid)
```

A grid has an analytic gradient, and so do the cheapest terms, such as
{class}`~pyrite.scoring.InternalOverlap`. A term with a slow, finite-difference gradient, such as
{class}`~pyrite.scoring.InternalEnergy`, makes the gradient of the whole scoring function slow:
the chain rule combines the gradients of all terms, so the slowest term sets the pace. See
{doc}`writing_scoring_functions` for adding an analytic gradient to a term of your own.

## Terms between ligand and receptor

Each of these terms finds, for every ligand atom, its `k` nearest receptor atoms within a
`cutoff`, and scores every pair. Terms on the same receptor share that search, at the largest `k`
and cutoff among them. Two things decide the cost:

- **The cutoff.** The wider it is, the more neighbours every atom has. The wide Gaussian of
  `vina_like` (`offset=3.0, width=2.0`) reaches about 12 Angstrom, and makes the shared search much
  more expensive than the other terms would on their own.
- **`k`.** The kernel works through all `k` neighbour slots of every atom, also the empty ones
  beyond the cutoff: a larger `k` is slower, even where there are few neighbours.

```{code-cell} python
from pyrite.scoring import Gaussian

terms = {
    "default": Gaussian(ligand, receptor),
    "wide": Gaussian(ligand, receptor, offset=3.0, width=2.0),
    "k=400": Gaussian(ligand, receptor, k=400),
}
for name, term in terms.items():
    print(f"{name:8s} {microseconds_per_pose(term):7.1f} us per pose")
```

On a grid, neither matters during the search: they only make the grid slower to build.

Only the atoms in a molecule's {attr}`~pyrite.Mol.scoring_mask` are scored, by default its heavy
atoms. Hydrogens added to a molecule still cost a little, as their positions are computed, but are
not scored.

## Batches

{meth}`~pyrite.scoring.ScoringFunction.batch_scores` scores many poses in one call. It saves the
cost of a call for every pose, and computes the positions of all poses at once. That pays off
where a call costs more than its work: cheap terms, and grids. For the terms between ligand and
receptor, whose work is per pair of atoms, a batch is about as fast as one pose at a time:

```{code-cell} python
from pyrite.scoring import InternalOverlap

terms = {"InternalOverlap": InternalOverlap(ligand), "grid": grid, "exact": exact}
for name, term in terms.items():
    single, batched = microseconds_per_pose(term), microseconds_per_pose(term, batched=True)
    print(f"{name:16s} one at a time {single:7.1f} us   batched {batched:7.1f} us per pose")
```

Batches are used where many poses are scored at once: filtering placements
({func}`~pyrite.search.boltzmann_diversity_filter`), or optimizers that work on a population. A
local search scores one pose at a time.

## Running in parallel

The searches from different starting poses, and the docking runs of different ligands, are
independent: they can run at the same time, one per core, with {mod}`concurrent.futures`. This
runs in a script, not a notebook: the worker processes import the function they run, so it must be
defined at the top level of a file, and the pool started under `if __name__ == "__main__":`.

```python
# dock.py
from concurrent.futures import ProcessPoolExecutor


def dock(ligand_file):
    ...  # load, place, search and cluster, as on the Searching page
    return best_poses


if __name__ == "__main__":
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(dock, ligand_files))
```

Let every process build what it needs, such as its scoring function and its basin hopping: some
parts, such as the hop of {func}`~pyrite.search.random_hop`, cannot be sent between processes.

numpy, scipy and numba can start threads of their own. With one process per core, set
`OMP_NUM_THREADS=1` (and `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`) before importing them, so that
the processes do not compete for the same cores.

## The size of the search

The number of poses a search scores is about the number of starting poses, times the hops per
start, times the scores per local minimization. A search that is too short misses good poses; one
that is too long wastes time. Fewer, better starting poses (see
{func}`~pyrite.search.boltzmann_diversity_filter`) and a step size that adapts
({func}`~pyrite.search.adaptive_stepsize`) make each score count. As every search is random,
how much is enough is best found by running a few known complexes with different seeds, and
seeing how often the search finds the known pose.
