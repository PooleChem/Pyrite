"""A ligand-receptor term restricts pairs with one hook, _mask: single, batch and grid must agree."""

import numpy as np
import pytest

from pyrite import Poses
from pyrite.atom_consts import AtomType
from pyrite.scoring.grid import GridScore
from pyrite.scoring.protein import _KNNScoringFunction

CARBON = [t for t in AtomType if t.name.startswith(("Aliphatic", "Aromatic"))]


class CarbonContact(_KNNScoringFunction):
    """Counts receptor atoms near the carbon atoms of the ligand, smoothly."""

    def _kernel(self, dist):
        return np.clip(1.0 - dist, 0.0, 1.0)

    def _mask(self, idx, mask, atom_type):
        return np.isin(atom_type, CARBON)[..., None] & mask


class Contact(CarbonContact):
    def _mask(self, idx, mask, atom_type):
        return mask


def test_single_and_batch_agree_with_a_mask(ctx):
    term = CarbonContact(ctx.ligand, ctx.receptor)
    single = np.array([term.get_score(pose) for pose in ctx.poses])
    np.testing.assert_allclose(term.batch_scores(ctx.poses), single, rtol=1e-12)


def test_the_mask_is_applied(ctx):
    pose = ctx.ligand.input_pose
    masked = CarbonContact(ctx.ligand, ctx.receptor).get_score(pose)
    unmasked = Contact(ctx.ligand, ctx.receptor).get_score(pose)
    assert 0 < masked < unmasked


def test_the_grid_applies_the_mask(ctx):
    term = CarbonContact(ctx.ligand, ctx.receptor)
    grid = GridScore(term, ctx.site, spacing=0.5)
    pose = ctx.ligand.input_pose
    assert grid.get_score(pose) == pytest.approx(term.get_score(pose), rel=0.05)
    poses = Poses.from_list([pose])
    assert grid.batch_scores(poses)[0] == pytest.approx(grid.get_score(pose))
