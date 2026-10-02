"""Conformance tests: every implementation of an extension point meets its contract.

The implementations come from the registries in ``conftest.py``; the checks are in
``contracts.py``. Adding an implementation there is all it takes to have it tested.
"""

import numpy as np
import pytest
from conftest import LAYOUTS
from contracts import (
    check_bounds,
    check_composition,
    check_hop,
    check_scoring_function,
    check_stepsize_rule,
    check_temperature_schedule,
    discover_subclasses,
)

import pyrite.bounds
import pyrite.scoring
from pyrite._common import Pose
from pyrite.bounds import Bounds
from pyrite.scoring import ScoringFunction

# ---------------------------------------------------------------------------
# Every public implementation is registered
# ---------------------------------------------------------------------------


def test_every_scoring_function_is_registered(scoring_registry):
    missing = [
        cls.__name__
        for cls in discover_subclasses(ScoringFunction, pyrite.scoring)
        if cls not in scoring_registry
    ]

    assert not missing, (
        f"Not registered for the conformance tests: {missing}. Add a line to SCORING_FACTORIES "
        "in tests/conftest.py."
    )


def test_every_bounds_is_registered(bounds_registry):
    missing = [
        cls.__name__
        for cls in discover_subclasses(Bounds, pyrite.bounds)
        if cls not in bounds_registry
    ]

    assert not missing, (
        f"Not registered for the conformance tests: {missing}. Add a line to BOUNDS_FACTORIES "
        "in tests/conftest.py."
    )


# ---------------------------------------------------------------------------
# The contracts, for every registered implementation
# ---------------------------------------------------------------------------


def test_scoring_function_contract(scoring_function, ctx):
    check_scoring_function(scoring_function, ctx.ligand, ctx.poses)


def test_scoring_function_composition_contract(scoring_function, ctx):
    check_composition(scoring_function, ctx.partner, ctx.ligand, ctx.poses)


def test_bounds_contract(bounds):
    check_bounds(bounds, LAYOUTS)


def test_hop_contract(hop_name, ctx):
    from conftest import HOP_FACTORIES

    check_hop(HOP_FACTORIES[hop_name](ctx), LAYOUTS)


def test_temperature_schedule_contract(schedule_name, ctx):
    from conftest import SCHEDULE_FACTORIES

    check_temperature_schedule(SCHEDULE_FACTORIES[schedule_name](ctx))


def test_stepsize_rule_contract(stepsize_rule_name, ctx):
    from conftest import STEPSIZE_FACTORIES

    check_stepsize_rule(STEPSIZE_FACTORIES[stepsize_rule_name](ctx))


# ---------------------------------------------------------------------------
# The checks themselves: they must reject implementations that break the contract
# ---------------------------------------------------------------------------


class _Constant(ScoringFunction):
    def _score(self, conf_id, computed):
        return 1.0


class _NotFinite(_Constant):
    def _score(self, conf_id, computed):
        return float("nan")


class _NotDeterministic(_Constant):
    calls = 0

    def _score(self, conf_id, computed):
        type(self).calls += 1
        return float(type(self).calls)


class _DependenciesAsSet(_Constant):
    def get_dependencies(self):
        return set()


class _WrongBatch(_Constant):
    def _batch_scores(self, conf_ids, computed_batch):
        return np.zeros(len(conf_ids))


class _LeaksConformers(_Constant):
    def step(self, x, mol):
        mol.update(x, new_conf=True)
        return 1.0


@pytest.mark.parametrize(
    "broken",
    [_NotFinite, _NotDeterministic, _DependenciesAsSet, _WrongBatch, _LeaksConformers],
)
def test_the_scoring_function_check_rejects_a_broken_implementation(broken, ctx):
    # The well-behaved baseline passes ...
    check_scoring_function(_Constant(), ctx.ligand, ctx.poses)
    # ... and each deliberate violation is caught.
    with pytest.raises(AssertionError):
        check_scoring_function(broken(), ctx.ligand, ctx.poses)


def test_the_composition_check_rejects_a_composite_that_changes_scores(ctx):
    class _ScalesWrongly(_Constant):
        def __mul__(self, other):
            return _Constant()  # 1.0 instead of the product

    check_composition(_Constant(), _Constant(), ctx.ligand, ctx.poses)
    with pytest.raises(AssertionError):
        check_composition(_ScalesWrongly(), _Constant(), ctx.ligand, ctx.poses)


def test_the_hop_check_rejects_a_broken_hop():
    def good(pose, rng, stepsize):
        return Pose(
            np.asarray(pose) + rng.normal(size=len(np.asarray(pose))) * stepsize, pose.layout
        )

    def modifies_input(pose, rng, stepsize):
        np.asarray(pose)[0] += 1.0
        return good(pose, rng, stepsize)

    def uses_global_random(pose, rng, stepsize):
        return Pose(np.asarray(pose) + np.random.normal(size=len(np.asarray(pose))), pose.layout)

    def returns_an_array(pose, rng, stepsize):
        return np.asarray(pose) + 1.0

    def never_moves(pose, rng, stepsize):
        return Pose(np.asarray(pose).copy(), pose.layout)

    check_hop(good, LAYOUTS)
    for broken in (modifies_input, uses_global_random, returns_an_array, never_moves):
        with pytest.raises(AssertionError):
            check_hop(broken, LAYOUTS)


def test_the_schedule_and_stepsize_checks_reject_broken_ones():
    check_temperature_schedule(lambda i, niter: 1.0)
    for broken in (
        lambda i, niter: -1.0,
        lambda i, niter: float("nan"),
        lambda i, niter: np.random.random(),
    ):
        with pytest.raises(AssertionError):
            check_temperature_schedule(broken)

    check_stepsize_rule(lambda stepsize, accepted, i: stepsize)
    for broken in (lambda stepsize, accepted, i: 0.0, lambda stepsize, accepted, i: float("inf")):
        with pytest.raises(AssertionError):
            check_stepsize_rule(broken)

    def mutates_history(stepsize, accepted, i):
        if len(accepted):
            accepted[0] = not accepted[0]
        return stepsize

    with pytest.raises(AssertionError):
        check_stepsize_rule(mutates_history)
