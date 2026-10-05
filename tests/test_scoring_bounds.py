"""The scoring functions built on `Bounds` and `Pocket`, and the `k = 1` nearest-neighbour query."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from pyrite.bounds import SphericalBounds
from pyrite.scoring import DistanceToPocket, KNNDependency, OutOfBoundsPenalty
from pyrite.scoring.dependencies import PositionQuery, Realization


def test_distance_to_pocket_equals_an_independent_computation(ctx):
    ligand, pocket = ctx.ligand, ctx.pocket
    scoring = DistanceToPocket(ligand, pocket)
    tree = cKDTree(pocket.centers)

    for pose in ctx.poses:
        distance = tree.query(ligand.pose_to_positions(pose))[0]
        expected = np.maximum(distance - pocket.radii[0], 0.0)
        expected[distance > scoring.cutoff] = scoring.cutoff
        expected[~scoring._mask] = 0.0

        assert scoring.get_score(pose) == pytest.approx(expected.sum(), abs=1e-9)


def test_distance_to_pocket_uses_every_atom(ctx):
    # A k = 1 query used to be narrowed to a single atom (the neighbour axis was missing).
    ligand, pocket = ctx.ligand, ctx.pocket
    scoring = DistanceToPocket(ligand, pocket)
    pose = ctx.poses[0]

    r, idx, mask = scoring._nn_dep.compute(Realization(pose, batched=False))

    assert r.shape == idx.shape == mask.shape == (ligand.n_atoms, 1)
    assert r.shape == scoring._nn_dep.narrow((r, idx, mask))[0].shape


def test_knn_dependency_keeps_the_neighbour_axis_for_every_k(ctx):
    ligand, receptor = ctx.ligand, ctx.receptor
    points = receptor.positions

    for k in (1, 2, 5):
        dependency = KNNDependency(points, PositionQuery(ligand), k, 8.0)
        r, idx, mask = dependency.compute(Realization(ctx.poses[0], batched=False))
        batch = dependency.compute(Realization(ctx.poses[:2], batched=True))[0]

        assert r.shape == idx.shape == mask.shape == (ligand.n_atoms, k)
        assert batch.shape == (2, ligand.n_atoms, k)
        assert np.array_equal(batch[0], r)
        assert dependency.narrow((r, idx, mask))[0].shape == r.shape


def test_out_of_bounds_penalty_is_zero_inside_and_grows_outside(ctx):
    ligand = ctx.ligand
    centre = tuple(ligand.get_positions().mean(axis=0))
    everything = OutOfBoundsPenalty(ligand, SphericalBounds(100.0, at=centre))
    small = OutOfBoundsPenalty(ligand, SphericalBounds(1.0, at=centre))
    smaller = OutOfBoundsPenalty(ligand, SphericalBounds(0.5, at=centre))

    assert everything.get_score(ligand.input_pose) == 0.0
    assert 0.0 < small.get_score(ligand.input_pose) < smaller.get_score(ligand.input_pose)
