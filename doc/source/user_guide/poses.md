---
file_format: mystnb
kernelspec:
  name: python3
---

# Poses

A docking search does not move a molecule's atoms one by one: it changes a pose. A
{class}`~pyrite.Pose` is a short vector that places a molecule in space: its rotation, its
translation, and the angle of every torsion. Scoring functions score poses, and searches change
them.

```{code-cell} python
import numpy as np

from pyrite import Mol, Pose, Poses

ligand = Mol.from_sdf("input/2boh_ligand.sdf", flexible=True, hydrogens="add")
pose = ligand.input_pose
np.asarray(pose)
```

## The parts of a pose

The {class}`~pyrite.PoseLayout` of a molecule says where every part is in the vector: three Euler
angles for the rotation, three coordinates for the translation, and one angle per torsion.

```{code-cell} python
layout = ligand.layout
layout.rot_slice, layout.trans_slice, layout.tors_slice, layout.n_dims
```

The parts are also available by name. The translation is the position of the center atom, and the
torsions are in radians:

```{code-cell} python
pose.rotation, pose.translation, pose.torsions
```

{attr}`~pyrite.Mol.input_pose` is the pose of the positions the molecule was loaded with: here the
crystal pose. Its rotation is zero, as the rotation is measured from those positions.

## From a pose to positions

{meth}`~pyrite.Mol.pose_to_positions` gives the positions of the atoms in a pose, without changing
the molecule:

```{code-cell} python
positions = ligand.pose_to_positions(pose)
float(np.abs(positions - ligand.positions).max())
```

A pose is a numpy array: change a copy of it to make a new pose. Moving a pose moves every atom:

```{code-cell} python
moved = Pose(np.asarray(pose).copy(), layout)
moved.translation += [3.0, 0.0, 0.0]
(ligand.pose_to_positions(moved) - positions).mean(axis=0)
```

Turning a torsion moves only the atoms on one side of its bond, the side away from the center atom:

```{code-cell} python
turned = Pose(np.asarray(pose).copy(), layout)
turned.torsions[0] += np.pi
moved_atoms = np.linalg.norm(ligand.pose_to_positions(turned) - positions, axis=1) > 1e-3
int(moved_atoms.sum()), ligand.n_atoms
```

## Many poses

{class}`~pyrite.Poses` is a batch of poses: an array with one pose per row. Scoring functions score a
whole batch at once (see {meth}`~pyrite.scoring.ScoringFunction.batch_scores`), and turning a batch
into positions is much faster than one pose at a time.

{meth}`~pyrite.Poses.from_parts` puts a batch together from its parts. Here, ten random rotations
around the crystal position, each with the torsions of an RDKit conformer:

```{code-cell} python
rng = np.random.default_rng(0)
n = 10
poses = Poses.from_parts(
    layout.sample_random_rotations(n, rng),
    np.tile(pose.translation, (n, 1)),
    ligand.get_n_conformer_torsion_configurations(n),
)
len(poses), np.asarray(poses).shape
```

Indexing a batch with a number gives a `Pose`, and with a slice or a list of indices a smaller
`Poses`. {meth}`~pyrite.Poses.from_list` turns a list of poses into a batch:

```{code-cell} python
type(poses[0]).__name__, len(poses[2:5]), len(Poses.from_list([pose, moved, turned]))
```

The positions of a batch have one more axis:

```{code-cell} python
ligand.pose_to_positions(poses).shape
```

A {class}`~pyrite.Viewer` steps through a batch with a slider:

```{code-cell} python
from pyrite import Viewer

Viewer(width=600, height=400).add_v(ligand, poses)
```

Starting poses for a search are usually not made by hand, but placed in a binding pocket by
{func}`~pyrite.search.place_in`: see the Searching page.

## Euler angles or quaternions

The rotation of a pose is three Euler angles by default. With `rotation_type="quat"`, it is a
quaternion `(w, x, y, z)` instead: four numbers, without the singularities of Euler angles, which
can help a local optimizer. The layout follows:

```{code-cell} python
q_ligand = Mol.from_sdf(
    "input/2boh_ligand.sdf", flexible=True, hydrogens="add", rotation_type="quat"
)
q_ligand.layout.n_dims, q_ligand.input_pose.rotation
```

A rigid molecule (not `flexible`) has no torsions: a pose is only a rotation and a translation.

## From positions to poses

{meth}`~pyrite.Mol.pose_from_positions` goes the other way: it finds the pose that puts the atoms at
given positions, such as those of a docked pose from another program:

```{code-cell} python
target = ligand.pose_to_positions(poses[3])
found = ligand.pose_from_positions(target)
float(np.abs(ligand.pose_to_positions(found) - target).max())
```

{meth}`~pyrite.Mol.to_sdf` writes poses to an SDF file, one record per pose, and
{meth}`~pyrite.Mol.poses_from_sdf` reads them back as poses of the molecule:

```{code-cell} python
ligand.to_sdf("poses.sdf", poses=poses)
read = ligand.poses_from_sdf("poses.sdf")
len(read)
```

```{code-cell} python
:tags: [remove-cell]

import os

os.remove("poses.sdf")
```
