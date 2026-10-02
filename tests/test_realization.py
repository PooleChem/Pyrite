"""How a pose becomes positions or an RDKit molecule while scoring: shared, private, and the
molecule being scored is never touched."""

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
    """Count the conformers made on the molecule and the numpy position computations."""
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
    assert ligand.n_conformers == 1


class _Records(ScoringFunction):
    """Remembers the posed RDKit molecules it was given, and scores the x of their centroid."""

    def __init__(self, mol):
        self.mol, self.seen = mol, []
        from pyrite.scoring.dependencies import RDKitDependency

        self.rdkit_dep = RDKitDependency(mol)

    def get_dependencies(self):
        return [self.rdkit_dep]

    def _score(self, pose, computed):
        self.seen.append(computed[self.rdkit_dep])
        return 1.0


def test_rdkit_terms_share_one_private_copy_per_pose(ctx, ligand):
    first, second = _Records(ligand), _Records(ligand)
    sf = first + second + InternalEnergy(ligand) + RMSD(ligand)

    sf.batch_scores(ctx.poses)

    assert len(first.seen) == len(second.seen) == len(ctx.poses)
    assert all(a is b for a, b in zip(first.seen, second.seen, strict=True))  # shared per pose
    assert len({id(m) for m in first.seen}) == len(ctx.poses)  # a different copy per pose
    assert all(m is not ligand.rdkit for m in first.seen)  # never the molecule being scored
    for posed, expected in zip(first.seen, ligand.pose_to_positions(ctx.poses), strict=True):
        assert posed.GetNumConformers() == 1
        assert np.allclose(posed.GetConformer().GetPositions(), expected)


def test_scoring_never_modifies_the_molecule_even_with_rdkit_terms(ctx, ligand):
    sf = (
        Gaussian(ligand, ctx.receptor)
        + 1e-2 * InternalEnergy(ligand)
        + InternalOverlap(ligand)
        + RMSD(ligand)
    )
    before = ligand.rdkit.ToBinary()
    update, positions = _spies(ligand)

    with update as update_spy, positions as positions_spy:
        sf.get_score(ctx.poses[0])
        sf.batch_scores(ctx.poses)

    assert update_spy.call_count == 0  # no conformer on the molecule, ever
    assert positions_spy.call_count == 2  # the positions are computed once per call
    assert ligand.rdkit.ToBinary() == before  # not a conformer, coordinate or property


def test_a_failing_term_leaves_the_molecule_untouched(ctx, ligand):
    class Explodes(ScoringFunction):
        def get_dependencies(self):
            return InternalEnergy(ligand).get_dependencies()

        def _score(self, pose, computed):
            raise RuntimeError("boom")

    before = ligand.rdkit.ToBinary()
    with pytest.raises(RuntimeError):
        Explodes().get_score(ctx.poses[0])

    assert ligand.rdkit.ToBinary() == before


def test_a_pose_of_another_layout_is_rejected(ctx, ligand):
    other = Mol.from_smiles("c1ccccc1O")

    with pytest.raises(AssertionError):
        Gaussian(ligand, ctx.receptor).get_score(Pose(np.zeros(other.layout.n_dims), other.layout))


def test_threads_can_score_poses_of_one_molecule_at_once(ctx, ligand):
    # Nothing is written to the molecule while scoring, so no private copy is needed. This
    # includes the terms that need RDKit: they get a private copy of the molecule per call.
    from concurrent.futures import ThreadPoolExecutor

    sf = (
        Gaussian(ligand, ctx.receptor)
        + 0.5 * Repulsion(ligand, ctx.receptor, k=100)
        + 1e-2 * InternalEnergy(ligand)
        + InternalOverlap(ligand)
        + RMSD(ligand)
    )
    poses = ctx.poses
    expected = np.array([sf.get_score(p) for p in poses])
    before = ligand.rdkit.ToBinary()

    with ThreadPoolExecutor(8) as pool:
        got = list(pool.map(sf.get_score, [poses[i % len(poses)] for i in range(200)]))

    assert np.allclose(got, np.resize(expected, 200))
    assert ligand.rdkit.ToBinary() == before


def test_a_new_rdkit_term_only_implements_score(ctx, ligand):
    from rdkit import Chem

    from pyrite.scoring._base import _RDKitScoringFunction

    class CentroidX(_RDKitScoringFunction):  # the whole implementation of a new term
        def _score(self, pose, computed):
            return float(
                Chem.rdMolTransforms.ComputeCentroid(computed[self.rdkit_dep].GetConformer()).x
            )

    term = CentroidX(ligand)
    sf = term + InternalEnergy(ligand) + RMSD(ligand)  # shares its posed copy with the others
    batch = sf.batch_scores(ctx.poses)

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
