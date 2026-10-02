"""How a pose becomes positions or a conformer while scoring: shared, minimal, always cleaned up."""

from unittest import mock

import numpy as np
import pytest

from pyrite import Mol, Pose
from pyrite.scoring import (
    RMSD,
    Gaussian,
    InternalEnergy,
    InternalOverlap,
    Repulsion,
    ScoringFunction,
)


@pytest.fixture
def ligand(ctx) -> Mol:
    return ctx.ligand


def _spies(ligand):
    """Count the RDKit conformers made and the numpy position computations."""
    update = mock.patch.object(
        Mol, "pose_to_conformer", autospec=True, side_effect=Mol.pose_to_conformer
    )
    positions = mock.patch.object(
        Mol, "pose_to_positions", autospec=True, side_effect=Mol.pose_to_positions
    )
    return update, positions


def test_knn_terms_never_touch_rdkit_and_compute_positions_once(ctx, ligand):
    sf = Gaussian(ligand, ctx.receptor) + 0.5 * Repulsion(ligand, ctx.receptor, k=100)
    update, positions = _spies(ligand)

    with update as update_spy, positions as positions_spy:
        sf.get_score(ctx.poses[0])
        sf.batch_scores(ctx.poses)

    assert update_spy.call_count == 0
    assert positions_spy.call_count == 2  # once per call, not once per term
    assert ligand.GetNumConformers() == 1


def test_rdkit_terms_share_one_conformer_per_pose_and_it_supplies_the_positions(ctx, ligand):
    sf = (
        Gaussian(ligand, ctx.receptor)
        + 1e-2 * InternalEnergy(ligand)
        + InternalOverlap(ligand)
        + RMSD(ligand)
    )
    update, positions = _spies(ligand)

    with update as update_spy, positions as positions_spy:
        sf.get_score(ctx.poses[0])
        assert update_spy.call_count == 1
        assert positions_spy.call_count == 0  # read off the conformer instead

        update_spy.reset_mock()
        sf.batch_scores(ctx.poses)
        assert update_spy.call_count == len(ctx.poses)
        assert positions_spy.call_count == 0

    assert ligand.GetNumConformers() == 1


def test_conformers_are_removed_when_scoring_fails(ctx, ligand):
    class Explodes(ScoringFunction):
        def get_dependencies(self):
            return InternalEnergy(ligand).get_dependencies()

        def _score(self, pose, computed):
            raise RuntimeError("boom")

    n = ligand.GetNumConformers()
    with pytest.raises(RuntimeError):
        Explodes().get_score(ctx.poses[0])

    assert ligand.GetNumConformers() == n


def test_a_pose_of_another_layout_is_rejected(ctx, ligand):
    other = Mol.from_smiles("c1ccccc1O")

    with pytest.raises(AssertionError):
        Gaussian(ligand, ctx.receptor).get_score(Pose(np.zeros(other.layout.n_dims), other.layout))


def test_threads_can_score_poses_of_one_molecule_at_once(ctx, ligand):
    # Nothing is written to the molecule while scoring KNN terms, so no private copy is needed.
    from concurrent.futures import ThreadPoolExecutor

    sf = Gaussian(ligand, ctx.receptor) + 0.5 * Repulsion(ligand, ctx.receptor, k=100)
    poses = ctx.poses
    expected = np.array([sf.get_score(p) for p in poses])
    conformers = ligand.GetNumConformers()

    with ThreadPoolExecutor(8) as pool:
        got = list(pool.map(sf.get_score, [poses[i % len(poses)] for i in range(200)]))

    assert np.allclose(got, np.resize(expected, 200))
    assert ligand.GetNumConformers() == conformers


def test_a_new_rdkit_term_only_implements_score(ctx, ligand):
    from rdkit import Chem

    from pyrite.scoring._base import _RDKitScoringFunction

    class CentroidX(_RDKitScoringFunction):  # the whole implementation of a new term
        def _score(self, pose, computed):
            return float(
                Chem.rdMolTransforms.ComputeCentroid(
                    self.mol.GetConformer(computed[self.rdkit_dep])
                ).x
            )

    term = CentroidX(ligand)
    sf = term + InternalEnergy(ligand) + RMSD(ligand)  # shares its conformer with the others
    update, _ = _spies(ligand)

    with update as update_spy:
        batch = sf.batch_scores(ctx.poses)
        assert update_spy.call_count == len(ctx.poses)  # one conformer per pose, not per term

    assert np.allclose(batch, [sf.get_score(p) for p in ctx.poses])
    centroids = np.array([ligand.pose_to_positions(p).mean(axis=0)[0] for p in ctx.poses])
    assert np.allclose(term.batch_scores(ctx.poses), centroids)  # the term itself is right


def test_the_default_batch_computes_the_dependencies_once_for_the_whole_batch(ctx, ligand):
    from pyrite.scoring.dependencies import KNNDependency

    class Custom(Gaussian):  # a KNN term that does not override the batched score
        _batch_scores = ScoringFunction._batch_scores

    sf = Custom(ligand, ctx.receptor)
    with mock.patch.object(
        KNNDependency, "compute", autospec=True, side_effect=KNNDependency.compute
    ) as spy:
        batch = sf.batch_scores(ctx.poses)

    assert spy.call_count == 1  # not once per pose
    assert np.allclose(batch, Gaussian(ligand, ctx.receptor).batch_scores(ctx.poses))
