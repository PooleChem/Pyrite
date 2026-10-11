"""vina_like is a shorthand: it must equal the scoring function written out term by term."""

import numpy as np
import pytest

from pyrite.scoring import (
    Gaussian,
    Hydrophobic,
    InternalEnergy,
    InternalOverlap,
    NonDirHBond,
    NumTors,
    Repulsion,
    vina_like,
    vina_like_grid,
)


def _written_out(ligand, receptor, internal):
    receptor_terms = (
        -0.035579 * Gaussian(ligand, receptor, offset=0.0, width=0.5)
        - 0.005156 * Gaussian(ligand, receptor, offset=3.0, width=2.0)
        + 0.840245 * Repulsion(ligand, receptor)
        - 0.035069 * Hydrophobic(ligand, receptor, good=0.5, bad=1.5)
        - 0.587439 * NonDirHBond(ligand, receptor, good=-0.7, bad=0.0)
    )
    score = receptor_terms if internal is None else receptor_terms + internal
    return score / (1 + 0.05846 * NumTors(ligand))


@pytest.mark.parametrize("internal", ["overlap", "energy", None])
def test_equals_the_terms_written_out(ctx, internal):
    ligand, receptor = ctx.ligand, ctx.receptor
    own = {
        "overlap": 0.5 * InternalOverlap(ligand),
        "energy": 0.01 * InternalEnergy(ligand),
        None: None,
    }[internal]
    expected = _written_out(ligand, receptor, own).batch_scores(ctx.poses)
    actual = vina_like(ligand, receptor, internal=internal).batch_scores(ctx.poses)
    np.testing.assert_allclose(actual, expected, rtol=1e-10)


def test_own_internal_term(ctx):
    own = 2.0 * InternalOverlap(ctx.ligand)
    expected = _written_out(ctx.ligand, ctx.receptor, own).get_score(ctx.ligand.input_pose)
    actual = vina_like(ctx.ligand, ctx.receptor, internal=own).get_score(ctx.ligand.input_pose)
    assert actual == pytest.approx(expected)


def test_grid_is_close_to_exact(ctx):
    exact = vina_like(ctx.ligand, ctx.receptor)
    grid = vina_like_grid(ctx.ligand, ctx.receptor, ctx.site, spacing=1.0)
    pose = ctx.ligand.input_pose
    assert grid.get_score(pose) == pytest.approx(exact.get_score(pose), abs=0.1)


def test_unknown_internal_raises(ctx):
    with pytest.raises(ValueError, match="Unknown internal term"):
        vina_like(ctx.ligand, ctx.receptor, internal="mmff")


def test_get_score_returns_python_floats(ctx):
    # Scores are plain floats, not numpy scalars, also in subscores (they show as np.float64(...)).
    score = vina_like(ctx.ligand, ctx.receptor)
    parts = {}
    assert type(score.get_score(ctx.ligand.input_pose, subscores=parts)) is float
    assert all(type(value) is float for value in parts.values())
    assert type(score.get_score(ctx.ligand.input_pose)) is float
