# Writing a scoring function

A scoring function turns a pose into a number: lower is better. Every scoring function in Pyrite is a
{class}`~pyrite.scoring.ScoringFunction`, and a new one only needs a single method. Everything else
is optional, and can be added later, when a scoring function turns out to be slow or is used often:

1. [The score of a pose](#the-score-of-a-pose): {meth}`~pyrite.scoring.ScoringFunction._score`, the only required method.
2. [Using RDKit](#using-rdkit): subclass {class}`~pyrite.scoring._base._RDKitScoringFunction`.
3. [A term between ligand and receptor](#a-term-between-ligand-and-receptor): subclass
   {class}`~pyrite.scoring.protein._KNNScoringFunction` and write a kernel.
4. [Sharing work](#sharing-work-with-dependencies): dependencies.
5. [Scoring many poses at once](#scoring-many-poses-at-once): {meth}`~pyrite.scoring.ScoringFunction._batch_scores`.
6. [Gradients](#gradients): {meth}`~pyrite.scoring.ScoringFunction._score_and_gradient`.
7. [Testing](#testing): the conformance tests.

Whatever level a scoring function is written at, it can be combined with all others
(`+`, `-`, `*`, `/`, `**`), scored one pose or many poses at a time, and used in a search with a
gradient.

## The score of a pose

Implement {meth}`_score(pose, computed) <pyrite.scoring.ScoringFunction._score>`, which returns the score of one pose as a float. The pose holds
the rotation, translation and torsions of the molecule; {meth}`~pyrite.Mol.pose_to_positions`
turns it into the positions of the atoms. For example, the radius of gyration:

```python
import numpy as np

from pyrite.scoring import ScoringFunction


class RadiusOfGyration(ScoringFunction):
    """Calculates the radius of gyration of a molecule."""

    def __init__(self, mol):
        self.mol = mol

    def _score(self, pose, computed):
        positions = self.mol.pose_to_positions(pose)
        centered = positions - positions.mean(axis=0)
        return float(np.sqrt((centered**2).sum(axis=1).mean()))
```

It can be used and combined like any other scoring function:

```python
compact = vina_like + 0.1 * RadiusOfGyration(ligand)
score = compact.get_score(pose)
scores = compact.batch_scores(poses)
score, gradient = compact.get_score_and_gradient(pose)
```

A scoring function must not change the molecule: {meth}`~pyrite.Mol.pose_to_positions` computes
the positions without touching it, so poses of one molecule can be scored from several threads at
once. `computed` holds the results of the dependencies of the scoring function (see
[Sharing work](#sharing-work-with-dependencies)); the simplest scoring functions do not use it.

## Using RDKit

To use an RDKit function on a pose, subclass
{class}`~pyrite.scoring._base._RDKitScoringFunction`. It gives {meth}`~pyrite.scoring.ScoringFunction._score` an RDKit molecule with the
pose as its only conformer, as `computed[self.rdkit_dep]`:

```python
from rdkit.Chem import Descriptors3D

from pyrite.scoring._base import _RDKitScoringFunction


class Asphericity(_RDKitScoringFunction):
    """Calculates the asphericity of a molecule."""

    def _score(self, pose, computed):
        posed = computed[self.rdkit_dep]
        return Descriptors3D.Asphericity(posed)
```

The RDKit molecule is a private copy, made once per pose and shared by all RDKit-based terms of a
composite: it may be changed freely. Outside of a scoring function, {meth}`~pyrite.Mol.to_rdkit`
gives the same copy.

## A term between ligand and receptor

Most terms of a docking scoring function score every ligand atom against the receptor atoms near
it. {class}`~pyrite.scoring.protein._KNNScoringFunction` does the work: it finds the `k` nearest
receptor atoms of every ligand atom within the cutoff, once for all terms on the same receptor, and
sums the score of every pair. A new term only implements {meth}`~pyrite.scoring.protein._KNNScoringFunction._kernel`, the score of a pair from its
distance beyond the optimal distance (the sum of the radii of the two atoms):

```python
import numpy as np

from pyrite.scoring.protein import _KNNScoringFunction


class Contact(_KNNScoringFunction):
    """Counts the receptor atoms in contact with the ligand, smoothly."""

    def _kernel(self, dist):
        # 1 up to the optimal distance, falling linearly to 0 one Angstrom beyond it
        return np.clip(1.0 - dist, 0.0, 1.0)
```

{meth}`~pyrite.scoring.protein._KNNScoringFunction._kernel` is called with the distances of all atoms, and of many poses at once, so write it with
numpy operations on the whole array. In return, the term is batched, and can be put on a grid with
{class}`~pyrite.scoring.grid.GridScore`, without any further code.

To score only some pairs, as {class}`~pyrite.scoring.Hydrophobic` scores only hydrophobic atoms,
also implement {meth}`_mask(idx, mask) <pyrite.scoring.protein._KNNScoringFunction._mask>`, which returns which neighbors count. Keep the cutoff and `k` in
mind: a term only sees the `k` nearest neighbors, so a term that reaches far needs a larger `k`.

## Sharing work with dependencies

An expensive computation that several terms need, such as the nearest neighbor search, is a
{class}`~pyrite.scoring.dependencies.Dependency`. A scoring function lists its dependencies in
{meth}`~pyrite.scoring.ScoringFunction.get_dependencies`, and reads their results from `computed`. Dependencies that compute the same
thing are merged, so the work is done once per pose, however many terms ask for it.

The positions of a molecule are available as a dependency too,
{class}`~pyrite.scoring.dependencies.PositionDependency`:

```python
from pyrite.scoring.dependencies import PositionDependency


class RadiusOfGyration(ScoringFunction):
    def __init__(self, mol):
        self.mol = mol
        self.position_dep = PositionDependency(mol)

    def get_dependencies(self):
        return [self.position_dep]

    def _score(self, pose, computed):
        positions = computed[self.position_dep]
        centered = positions - positions.mean(axis=0)
        return float(np.sqrt((centered**2).sum(axis=1).mean()))
```

A new kind of dependency implements {meth}`~pyrite.scoring.dependencies.Dependency.compute`, {meth}`~pyrite.scoring.dependencies.Dependency.group_key` (dependencies with the same key are
merged) and {meth}`~pyrite.scoring.dependencies.Dependency.merge_group`; see {class}`~pyrite.scoring.dependencies.Dependency`.

## Scoring many poses at once

{meth}`~pyrite.scoring.ScoringFunction.batch_scores` scores many poses at once. By default it calls
{meth}`~pyrite.scoring.ScoringFunction._score` for every pose, on dependencies computed once for the whole batch. A scoring function that
can do better implements {meth}`_batch_scores(poses, computed_batch) <pyrite.scoring.ScoringFunction._batch_scores>`, where every dependency has a
leading axis of the number of poses:

```python
    def _batch_scores(self, poses, computed_batch):
        positions = computed_batch[self.position_dep]  # (n_poses, n_atoms, 3)
        centered = positions - positions.mean(axis=1, keepdims=True)
        return np.sqrt((centered**2).sum(axis=2).mean(axis=1))
```

## Gradients

A local optimizer, such as the L-BFGS-B in {class}`~pyrite.search.BasinHopping`, needs the
gradient of the score with respect to the pose. By default it is computed by finite differences:
two scores for every variable of the pose. A scoring function that knows how its score changes when
an atom moves (`dS/dx`, the "force" on every atom) implements {meth}`~pyrite.scoring.ScoringFunction._score_and_gradient`, and lets
{meth}`~pyrite.Mol.pose_gradient` turn the forces into the gradient with respect to the pose:

```python
    def _score_and_gradient(self, pose, computed):
        positions = computed[self.position_dep]
        centered = positions - positions.mean(axis=0)
        radius = np.sqrt((centered**2).sum(axis=1).mean())
        forces = centered / (len(positions) * radius)  # dS/dx of every atom
        return float(radius), self.mol.pose_gradient(pose, positions, forces)
```

The gradient of a composite follows from the chain rule, so terms with and without an analytic
gradient can be combined.

## Testing

The conformance tests check every scoring function in Pyrite automatically: finite and
deterministic scores, {meth}`~pyrite.scoring.ScoringFunction.batch_scores` equal to scoring the poses one by one, a molecule that is left
untouched, a gradient equal to finite differences, and the same results when combined with other
terms. A new scoring function in {mod}`pyrite.scoring` is registered with one line in
`SCORING_FACTORIES`, in `tests/conftest.py` (the tests are in the source repository, not in the
installed package):

```python
SCORING_FACTORIES = {
    ...
    RadiusOfGyration: lambda c: RadiusOfGyration(c.ligand),
}
```

A public scoring function that is not registered makes the tests fail, so it cannot be forgotten.
