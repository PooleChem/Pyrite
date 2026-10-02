"""Conformance checks: what every implementation of an extension point must satisfy.

Each ``check_*`` function states a contract in its docstring and asserts it. They are called
for every registered implementation by ``test_contracts.py`` (see ``conftest.py`` to register
a new one), and can be called directly on your own implementation:

    check_scoring_function(MyScore(ligand, receptor), ligand, poses)

The contracts are deliberately minimal: invariants that hold *by definition* of the extension
point, not properties of today's implementations. Do not add a check that a valid new
implementation could reasonably violate (for example "scores are non-negative").
"""

from __future__ import annotations

import contextlib
import numbers

import numpy as np
from helpers import rdkit_positions

from pyrite._common import Pose, PoseLayout
from pyrite.scoring import Dependency, ScoringFunction

RTOL, ATOL = 1e-7, 1e-8

# Modules that cannot be imported and are not part of the public API (dead code).
KNOWN_BROKEN_MODULES = {"pyrite.scoring.openmm"}


def discover_subclasses(base: type, package) -> list[type]:
    """Find the public, concrete subclasses of `base` defined in `package`.

    Every submodule of the package is imported first, so a new module is found without being
    mentioned anywhere. Classes whose name starts with an underscore, abstract classes, and
    classes defined outside the package (for example in a test) are skipped.
    """
    import importlib
    import inspect
    import pkgutil

    for info in pkgutil.walk_packages(getattr(package, "__path__", []), package.__name__ + "."):
        if info.name not in KNOWN_BROKEN_MODULES:
            importlib.import_module(info.name)

    found, stack = {}, list(base.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls not in found:
            found[cls] = None
            stack += cls.__subclasses__()
    return sorted(
        (
            cls
            for cls in found
            if not cls.__name__.startswith("_")
            and cls.__module__.startswith(package.__name__)
            and not inspect.isabstract(cls)
        ),
        key=lambda cls: cls.__name__,
    )


def _close(got, expected) -> bool:
    return bool(np.allclose(got, expected, rtol=RTOL, atol=ATOL))


@contextlib.contextmanager
def _positions_through_rdkit(mol):
    """Compute the positions of poses of `mol` with RDKit (``helpers.rdkit_positions``) instead of numpy.

    The reference the numpy pipeline must agree with, for whatever a scoring function does with
    those positions.
    """
    cls = type(mol)
    original = cls.pose_to_positions

    def through_rdkit(self, poses):
        if self is not mol:
            return original(self, poses)
        return rdkit_positions(self, np.asarray(poses, dtype=float))

    cls.pose_to_positions = through_rdkit
    try:
        yield
    finally:
        cls.pose_to_positions = original


# ---------------------------------------------------------------------------
# ScoringFunction
# ---------------------------------------------------------------------------


def check_scoring_function(sf: ScoringFunction, mol, poses) -> None:
    """Check the contract of a :class:`~pyrite.scoring.ScoringFunction`.

    - ``get_score(pose)`` returns a finite real number, and the same one when called again, also
      for the raw values of the pose.
    - ``batch_scores(poses)`` returns one score per pose, equal to calling ``get_score`` for each
      (in any order, and for raw values).
    - The score does not depend on how the positions of the pose were computed (numpy, or RDKit).
    - Scoring leaves `mol` as it was: no extra conformers, nothing moved, nothing changed.
    - ``get_dependencies()`` returns a list of :class:`~pyrite.scoring.Dependency`, which are
      hashable and equal to themselves.

    Parameters
    ----------
    sf : ScoringFunction
        The scoring function, bound to `mol`.
    mol : Mol
        The molecule whose poses are scored. It is not modified.
    poses : Poses
        A few poses of `mol`, in its layout.
    """
    assert isinstance(sf, ScoringFunction), f"{type(sf).__name__} is not a ScoringFunction"
    n_conformers, loaded = mol.n_conformers, mol.get_positions().copy()
    state = mol.rdkit.ToBinary()

    scores = np.array([sf.get_score(pose) for pose in poses])
    assert all(isinstance(s, numbers.Real) for s in scores), "get_score must return a float"
    assert np.isfinite(scores).all(), f"non-finite scores: {scores}"
    assert _close([sf.get_score(p) for p in poses], scores), "get_score is not deterministic"
    raw = np.asarray(poses)
    assert _close([sf.get_score(row) for row in raw], scores), "raw values score differently"

    batch = np.asarray(sf.batch_scores(poses))
    assert batch.shape == (len(poses),), f"batch_scores returned shape {batch.shape}"
    assert _close(batch, scores), f"batch_scores {batch} differs from get_score {scores}"
    reversed_ = np.asarray(sf.batch_scores(poses[::-1]))
    assert _close(reversed_, scores[::-1]), "batch_scores depends on the order of the poses"
    assert _close(sf.batch_scores(raw), scores), "batch_scores of raw values differs"

    with _positions_through_rdkit(mol):
        assert _close([sf.get_score(p) for p in poses], scores), (
            "get_score depends on how the positions are computed"
        )
        assert _close(sf.batch_scores(poses), scores), (
            "batch_scores depends on how the positions are computed"
        )

    assert mol.n_conformers == n_conformers, "scoring left conformers on the molecule"
    assert np.array_equal(mol.get_positions(), loaded), "scoring moved the molecule"
    assert mol.rdkit.ToBinary() == state, "scoring changed the RDKit molecule"

    dependencies = sf.get_dependencies()
    assert isinstance(dependencies, list), (
        f"get_dependencies() must return a list, not {type(dependencies).__name__}: a set "
        "silently merges distinct dependencies that compare equal"
    )
    for dependency in dependencies:
        assert isinstance(dependency, Dependency)
        hash(dependency)
        assert dependency == dependency  # noqa: PLR0124


def check_composition(sf: ScoringFunction, partner: ScoringFunction, mol, poses) -> None:
    """Check that combining scoring functions does not change what they compute.

    A composite shares and merges the dependencies of its terms (one nearest-neighbour query
    for several terms, say), and that must be invisible. For every operator below, both
    ``get_score`` and ``batch_scores`` of the composite must equal the same arithmetic on the
    scores of the parts.

    Parameters
    ----------
    sf : ScoringFunction
        The scoring function under test.
    partner : ScoringFunction
        Another scoring function on the same molecule(s) to combine it with. Best if it
        shares dependencies with `sf` with a different `k` or cutoff.
    mol : Mol
        The molecule. It is not modified.
    poses : Poses
        A few poses of `mol`.
    """
    a = np.array([sf.get_score(p) for p in poses])
    b = np.array([partner.get_score(p) for p in poses])
    lo, hi = np.quantile(a, 0.25), np.quantile(a, 0.75)
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0

    cases = {
        "a + b": (sf + partner, a + b),
        "a - b": (sf - partner, a - b),
        "a * b": (sf * partner, a * b),
        "-a": (-sf, -a),
        "2.5 * a": (2.5 * sf, 2.5 * a),
        "a * 2.5": (sf * 2.5, a * 2.5),
        "a / 4": (sf / 4.0, a / 4.0),
        "a + 1.5": (sf + 1.5, a + 1.5),
        "1.5 + a": (1.5 + sf, a + 1.5),
        "1.5 - a": (1.5 - sf, 1.5 - a),
        "a ** 2": (sf**2, a**2),
        "a + b + a": (sf + partner + sf, 2 * a + b),
        "(a + b) * (a - b)": ((sf + partner) * (sf - partner), (a + b) * (a - b)),
        "clamp(a)": (sf.clamp(lo, hi), np.clip(a, lo, hi)),
        "clamp(a + b)": ((sf + partner).clamp(max_score=hi), np.minimum(a + b, hi)),
    }
    for name, (composite, expected) in cases.items():
        got = np.array([composite.get_score(p) for p in poses])
        assert _close(got, expected), f"{name}: get_score {got} != {expected}"
        batch = np.asarray(composite.batch_scores(poses))
        assert _close(batch, expected), f"{name}: batch_scores {batch} != {expected}"


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


def check_bounds(bounds, layouts: list[PoseLayout]) -> None:
    """Check the contract of a :class:`~pyrite.bounds.Bounds`.

    - ``get_translation_bounds()`` gives three ``(min, max)`` pairs.
    - ``place_random_uniform(n, rng)`` returns finite positions of shape ``(n, 3)`` that lie
      within the bounds (``is_within``), reproducibly from the `rng`, and ``(0, 3)`` for ``n=0``.
    - ``place_grid(n)`` returns ``n ** 3`` finite positions within the translation bounds. A
      ``Bounds`` that cannot make a grid may raise ``NotImplementedError`` instead.
    - ``squared_distance(p)`` is zero for a point within the bounds, and positive, growing with
      the distance, for points outside (also optional: ``NotImplementedError``).
    - ``get_bounds(layout)`` has one ``(min, max)`` per variable of the layout, and its
      translation part is ``get_translation_bounds()``.

    Parameters
    ----------
    bounds : Bounds
        The bounds.
    layouts : list[PoseLayout]
        The layouts to check ``get_bounds`` for.
    """
    translation = bounds.get_translation_bounds()
    assert len(translation) == 3 and all(lo <= hi for lo, hi in translation)
    lo = np.array([b[0] for b in translation]) - 1e-9
    hi = np.array([b[1] for b in translation]) + 1e-9

    positions = bounds.place_random_uniform(40, rng=np.random.default_rng(0))
    assert positions.shape == (40, 3) and np.isfinite(positions).all()
    assert all(bounds.is_within(tuple(p)) for p in positions), "positions outside the bounds"
    assert ((positions >= lo) & (positions <= hi)).all()
    again = bounds.place_random_uniform(40, rng=np.random.default_rng(0))
    other = bounds.place_random_uniform(40, rng=np.random.default_rng(1))
    assert np.array_equal(positions, again), "place_random_uniform is not reproducible from rng"
    assert not np.array_equal(positions, other)
    assert bounds.place_random_uniform(0, rng=np.random.default_rng(0)).shape == (0, 3)

    try:
        grid = bounds.place_grid(3)
    except NotImplementedError:
        pass
    else:
        assert grid.shape == (27, 3) and np.isfinite(grid).all()
        assert ((grid >= lo) & (grid <= hi)).all()

    try:
        inside = [bounds.squared_distance(tuple(p)) for p in positions[:10]]
        assert all(d == 0 for d in inside), f"squared_distance is not zero within: {inside}"
        # Walk away from the bounds, along +x from the edge of the translation bounds.
        edge = np.array([hi[0], (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2])
        outside = [bounds.squared_distance(tuple(edge + [step, 0, 0])) for step in (1.0, 3.0, 9.0)]
        assert not bounds.is_within(tuple(edge + [1.0, 0, 0]))
        assert 0 < outside[0] < outside[1] < outside[2], f"not growing outside: {outside}"
    except NotImplementedError:
        pass

    for layout in layouts:
        variables = bounds.get_bounds(layout)
        assert len(variables) == layout.n_dims
        assert variables[layout.trans_slice] == translation


# ---------------------------------------------------------------------------
# Search: hops, temperature schedules, step size rules
# ---------------------------------------------------------------------------


def check_hop(hop, layouts: list[PoseLayout]) -> None:
    """Check the contract of a hop, ``hop(pose, rng, stepsize) -> pose``.

    - It returns a new :class:`~pyrite.Pose` in the layout of its input, with finite values.
    - It does not modify its input.
    - It draws all its randomness from `rng`: the same generator state gives the same result.
    - It does move the pose.

    A hop may ignore `stepsize`.

    Parameters
    ----------
    hop : callable
        The hop.
    layouts : list[PoseLayout]
        The layouts to check it for.
    """
    for layout in layouts:
        rng = np.random.default_rng(0)
        pose = Pose(
            np.concatenate(
                [
                    layout.sample_random_rotations(1, rng)[0],
                    rng.uniform(-3, 3, 3),
                    layout.sample_random_torsions(1, rng)[0],
                ]
            ),
            layout,
        )
        before = np.asarray(pose).copy()

        new = hop(pose, np.random.default_rng(1), 0.3)

        assert isinstance(new, Pose) and new.layout == layout, f"{layout}: not a Pose in the layout"
        assert new is not pose and np.isfinite(np.asarray(new)).all()
        assert np.array_equal(np.asarray(pose), before), "the hop modified its input"
        again = hop(pose, np.random.default_rng(1), 0.3)
        assert new == again, "the hop is not reproducible from the rng"
        moved = [hop(pose, np.random.default_rng(s), 0.3) != pose for s in range(5)]
        assert any(moved), "the hop never moves the pose"


def check_temperature_schedule(schedule) -> None:
    """Check the contract of a temperature schedule, ``T(i, niter) -> float``.

    It returns a finite, non-negative number for every hop ``i < niter``, and the same one
    when called again.

    Parameters
    ----------
    schedule : callable
        The schedule.
    """
    for niter in (1, 2, 50):
        values = [schedule(i, niter) for i in range(niter)]
        assert all(isinstance(v, numbers.Real) for v in values)
        assert np.isfinite(values).all() and (np.asarray(values) >= 0).all(), values
        assert values == [schedule(i, niter) for i in range(niter)], "not deterministic"


def check_stepsize_rule(adapt) -> None:
    """Check the contract of a step size rule, ``adapt(stepsize, accepted, i) -> stepsize``.

    It returns a finite, positive number; does not modify the acceptance history; always gives
    the same answer for the same input; and handles the first hop (empty history).

    Parameters
    ----------
    adapt : callable
        The rule.
    """
    rng = np.random.default_rng(0)
    for i in (0, 1, 5, 20, 40, 100):
        accepted = rng.random(i) < 0.5
        for stepsize in (0.05, 0.5, 2.0):
            history = accepted.copy()
            new = adapt(stepsize, history, i)
            assert np.array_equal(history, accepted), "the rule modified the acceptance history"
            assert isinstance(new, numbers.Real) and np.isfinite(new) and new > 0
            assert new == adapt(stepsize, accepted.copy(), i), "not deterministic"
