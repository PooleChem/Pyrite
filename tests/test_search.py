"""Regression tests for `pyrite.search`: the hop, the schedules, BasinHopping, the diversity filter."""

import warnings

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from pyrite._common import Pose, PoseLayout, Poses
from pyrite._util import _rotation_matrix_from_euler, _rotation_matrix_from_quat
from pyrite.search import (
    BasinHopping,
    adaptive_stepsize,
    boltzmann_diversity_filter,
    geometric_annealing,
    random_hop,
)

LAYOUT = PoseLayout("euler", 4)  # 3 + 3 + 4 = 10 variables
BOUNDS = np.array([[-2.0, 2.0]] * 3)


def _objective(pose) -> float:
    """A cheap multi-well objective on the raw vector."""
    x = np.asarray(pose)
    return float(np.sum(np.cos(3 * x) + 0.1 * x**2) + 0.5 * np.sin(5 * x[0] * x[3]))


def _start(seed: int = 3) -> Pose:
    return Pose(np.random.default_rng(seed).uniform(-1, 1, LAYOUT.n_dims), LAYOUT)


def _matrix(layout: PoseLayout, rotation: np.ndarray) -> np.ndarray:
    f = _rotation_matrix_from_euler if layout.rot_type == "euler" else _rotation_matrix_from_quat
    return f(*rotation)[:3, :3]


# ---------------------------------------------------------------------------
# PoseLayout.compose_rotation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rot_type", ["euler", "quat"])
def test_compose_rotation_is_a_left_multiplication(rot_type):
    layout = PoseLayout(rot_type, 0)
    rng = np.random.default_rng(0)
    rotations = layout.sample_random_rotations(20, rng)
    deltas = rng.normal(size=(20, 3)) * 0.7

    composed = layout.compose_rotation(rotations, deltas)

    for r, d, c in zip(rotations, deltas, composed, strict=True):
        expected = Rotation.from_rotvec(d).as_matrix() @ _matrix(layout, r)
        assert np.allclose(_matrix(layout, c), expected, atol=1e-12)
        assert np.allclose(_matrix(layout, layout.compose_rotation(r, d)), expected, atol=1e-12)


@pytest.mark.parametrize("rot_type", ["euler", "quat"])
def test_compose_rotation_with_zero_is_the_identity(rot_type):
    layout = PoseLayout(rot_type, 0)
    r = layout.sample_random_rotations(1, np.random.default_rng(1))[0]

    same = layout.compose_rotation(r, np.zeros(3))

    assert np.allclose(_matrix(layout, same), _matrix(layout, r), atol=1e-12)


# ---------------------------------------------------------------------------
# random_hop
# ---------------------------------------------------------------------------


def test_hop_returns_a_new_pose_and_keeps_the_input():
    pose = _start()
    before = np.asarray(pose).copy()

    new = random_hop(BOUNDS)(pose, np.random.default_rng(0), 0.3)

    assert isinstance(new, Pose) and new.layout == LAYOUT
    assert np.array_equal(np.asarray(pose), before)
    assert not np.array_equal(np.asarray(new), before)


def test_hop_clips_the_translation_to_the_bounds():
    hop, rng = random_hop(BOUNDS), np.random.default_rng(0)
    edge = Pose(np.concatenate([np.zeros(3), [1.9, -1.9, 0.0], np.zeros(4)]), LAYOUT)

    translations = np.stack([hop(edge, rng, 5.0).translation for _ in range(200)])

    assert (translations >= -2.0).all() and (translations <= 2.0).all()
    assert (translations == 2.0).any()  # a large step really is clipped, not just rare


def test_hop_stepsize_scales_the_torsion_step():
    hop, rng = random_hop(), np.random.default_rng(0)
    pose = Pose(np.zeros(LAYOUT.n_dims), LAYOUT)

    def mean_step(stepsize):
        return np.mean([np.abs(hop(pose, rng, stepsize).torsions).mean() for _ in range(500)])

    # Uniform in [-s, s]: the mean absolute step is s / 2.
    assert mean_step(0.1) == pytest.approx(0.05, rel=0.1)
    assert mean_step(1.0) == pytest.approx(0.5, rel=0.1)


@pytest.mark.parametrize("rot_type", ["euler", "quat"])
@pytest.mark.parametrize("pitch", [0.0, 1.5])  # 1.5 rad: next to the Euler gimbal lock
def test_hop_rotation_step_does_not_depend_on_the_orientation(rot_type, pitch):
    layout = PoseLayout(rot_type, 0)
    start = np.concatenate([[0.4, pitch, -0.8], np.zeros(3)])
    if rot_type == "quat":
        start = np.concatenate(
            [Rotation.from_euler("ZYX", [-0.8, pitch, 0.4]).as_quat()[[3, 0, 1, 2]], np.zeros(3)]
        )
    hop, rng = random_hop(), np.random.default_rng(0)
    pose = Pose(start, layout)

    def rotation(p):
        r = np.asarray(p.rotation)
        if rot_type == "quat":
            return Rotation.from_quat(r[[1, 2, 3, 0]])
        return Rotation.from_euler("ZYX", r[::-1])

    angles = [
        (rotation(hop(pose, rng, 0.05)) * rotation(pose).inv()).magnitude() for _ in range(3000)
    ]

    # A Gaussian rotation vector of scale s: the mean angle is s * E|N(0, I3)| = s * sqrt(8 / pi).
    assert np.mean(angles) == pytest.approx(0.05 * np.sqrt(8 / np.pi), rel=0.05)


def test_hop_keeps_quaternions_normalised_and_handles_rigid_molecules():
    quat = PoseLayout("quat", 2)
    pose = Pose(np.concatenate([quat.identity_rotation, np.zeros(3), np.zeros(2)]), quat)
    hop, rng = random_hop(), np.random.default_rng(0)

    assert np.allclose(np.linalg.norm(hop(pose, rng, 0.5).rotation), 1.0)
    rigid = PoseLayout("euler", 0)
    assert hop(Pose(np.zeros(6), rigid), rng, 0.3).torsions.shape == (0,)


def test_hop_is_reproducible_from_the_rng():
    hop, pose = random_hop(), _start()

    a = hop(pose, np.random.default_rng(5), 0.4)
    b = hop(pose, np.random.default_rng(5), 0.4)
    c = hop(pose, np.random.default_rng(6), 0.4)

    assert a == b and not a == c


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


def test_geometric_annealing():
    T = geometric_annealing(2.0, 0.1)

    assert T(0, 50) == 2.0
    assert T(49, 50) == pytest.approx(0.1)
    assert T(0, 1) == 2.0
    values = np.array([T(i, 50) for i in range(50)])
    assert (np.diff(values) < 0).all()
    assert np.allclose(values[1:] / values[:-1], values[1] / values[0])  # a constant ratio


def test_adaptive_stepsize_rule():
    adapt = adaptive_stepsize(target_rate=0.4, window=4, lo=0.05, hi=1.0)

    def accepted(*values):
        return np.array(values, dtype=bool)

    assert adapt(0.5, accepted(1, 1, 1, 1), 4) == pytest.approx(0.625)  # too many accepted: grow
    assert adapt(0.5, accepted(0, 0, 0, 0), 4) == pytest.approx(0.4)  # too few: shrink
    assert adapt(0.5, accepted(1, 0, 1, 0), 4) == 0.5  # inside the dead band
    assert adapt(0.5, accepted(1, 1, 1, 1), 5) == 0.5  # only at a window boundary
    assert adapt(0.5, accepted(1, 1, 1, 1), 0) == 0.5  # never before the first hop
    assert adapt(0.5, accepted(1, 1), 4) == 0.5  # not with too short a history
    assert adapt(0.9, accepted(1, 1, 1, 1), 4) == 1.0  # capped at hi
    assert adapt(0.06, accepted(0, 0, 0, 0), 4) == 0.05  # floored at lo


# ---------------------------------------------------------------------------
# BasinHopping
# ---------------------------------------------------------------------------

MINIMIZER = {"method": "L-BFGS-B", "options": {"eps": 1e-2}, "tol": 1e-4}


def _reference_run(func, x0, niter, T, stepsize, hop, adapt, rng):
    """The algorithm, written out from its description: this pins the semantics of the loop."""
    layout = x0.layout

    def minimize_from(pose):
        res = minimize(lambda v: func(Pose(v, layout)), np.asarray(pose), **MINIMIZER)
        return Pose(res.x.copy(), layout), res.fun

    x, fx = minimize_from(x0)
    best_x, best_fx = x, fx
    out = {k: [] for k in ("accepted", "stepsizes", "temperatures", "scores", "poses")}
    for i in range(niter):
        if adapt is not None:
            stepsize = adapt(stepsize, np.array(out["accepted"], dtype=bool), i)
        x_new, fx_new = minimize_from(hop(x, rng, stepsize))
        temperature = T(i, niter) if callable(T) else T
        accept = fx_new - fx < 0 or rng.random() < np.exp(-(fx_new - fx) / max(temperature, 1e-6))
        out["accepted"].append(accept)
        out["stepsizes"].append(stepsize)
        out["temperatures"].append(temperature)
        out["scores"].append(fx_new)
        out["poses"].append(np.asarray(x_new))
        if accept:
            x, fx = x_new, fx_new
        if fx_new < best_fx:
            best_x, best_fx = x_new, fx_new
    return best_x, best_fx, out


@pytest.mark.parametrize(
    ("T", "adapt"),
    [
        (0.5, None),
        (1e-9, None),
        (geometric_annealing(2.0, 0.1), adaptive_stepsize(window=5)),
    ],
    ids=["fixed", "cold", "annealed-adaptive"],
)
def test_basin_hopping_follows_the_metropolis_loop(T, adapt):
    niter, stepsize, hop = 30, 0.6, random_hop(BOUNDS)
    best_x, best_fx, ref = _reference_run(
        _objective, _start(), niter, T, stepsize, hop, adapt, np.random.default_rng(11)
    )

    result = BasinHopping(
        _objective,
        hop,
        T=T,
        stepsize=stepsize,
        adapt_stepsize=adapt,
        rng=np.random.default_rng(11),
    ).run(_start(), niter)

    assert isinstance(result.x, Pose) and result.x.layout == LAYOUT
    assert result.x == best_x and result.fun == best_fx
    assert np.array_equal(result.accepted, ref["accepted"])
    assert np.array_equal(result.stepsizes, ref["stepsizes"])
    assert np.array_equal(result.temperatures, ref["temperatures"])
    assert np.array_equal(result.scores, ref["scores"])
    assert np.array_equal(np.asarray(result.poses), np.stack(ref["poses"]))


def test_basin_hopping_result_is_consistent():
    niter = 20
    bh = BasinHopping(
        _objective, random_hop(BOUNDS), T=1.0, stepsize=0.5, rng=np.random.default_rng(2)
    )
    start = _start()
    start_min = minimize(lambda v: _objective(v), np.asarray(start), **MINIMIZER).fun

    result = bh.run(start, niter)

    for name in ("accepted", "stepsizes", "temperatures", "scores"):
        assert result[name].shape == (niter,)
    assert isinstance(result.poses, Poses) and result.poses.layout == LAYOUT
    assert result.nit == niter
    # Every row of `poses` really has the score recorded for it, and the best is the best seen.
    assert np.allclose([_objective(p) for p in result.poses], result.scores)
    assert result.fun == pytest.approx(min(start_min, result.scores.min()))
    assert (result.stepsizes == 0.5).all() and (result.temperatures == 1.0).all()


def test_basin_hopping_never_returns_something_worse_than_its_start():
    start = _start()
    start_min = minimize(lambda v: _objective(v), np.asarray(start), **MINIMIZER).fun

    result = BasinHopping(
        _objective, random_hop(BOUNDS), T=0.5, stepsize=0.6, rng=np.random.default_rng(0)
    ).run(start, 15)

    assert result.fun <= start_min + 1e-12


def test_basin_hopping_cold_never_accepts_a_worse_minimum():
    result = BasinHopping(
        _objective, random_hop(BOUNDS), T=1e-9, stepsize=0.6, rng=np.random.default_rng(0)
    ).run(_start(), 25)

    assert result.accepted.sum() < len(result.accepted)  # it does reject
    # Walk the chain: an accepted hop must have improved on the current score.
    start_min = minimize(lambda v: _objective(v), np.asarray(_start()), **MINIMIZER).fun
    current = start_min
    for accepted, score in zip(result.accepted, result.scores, strict=True):
        if accepted:
            assert score < current
            current = score


def test_basin_hopping_with_zero_temperature_is_greedy_and_does_not_divide_by_zero():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a division by zero would raise RuntimeWarning
        result = BasinHopping(
            _objective, random_hop(BOUNDS), T=0.0, stepsize=0.6, rng=np.random.default_rng(0)
        ).run(_start(), 15)

    current = minimize(lambda v: _objective(v), np.asarray(_start()), **MINIMIZER).fun
    for accepted, score in zip(result.accepted, result.scores, strict=True):
        if accepted:
            assert score < current
            current = score


def test_basin_hopping_hot_accepts_everything():
    result = BasinHopping(
        _objective, random_hop(BOUNDS), T=1e9, stepsize=0.6, rng=np.random.default_rng(0)
    ).run(_start(), 25)

    assert result.accepted.all()


def test_basin_hopping_is_reproducible_and_leaves_the_start_alone():
    start = _start()
    before = np.asarray(start).copy()

    def run(seed):
        return BasinHopping(
            _objective,
            random_hop(BOUNDS),
            T=geometric_annealing(1.0, 0.1),
            stepsize=0.6,
            adapt_stepsize=adaptive_stepsize(window=5),
            rng=np.random.default_rng(seed),
        ).run(start, 20)

    a, b, c = run(1), run(1), run(2)

    assert a.x == b.x and np.array_equal(a.accepted, b.accepted)
    assert not np.array_equal(a.accepted, c.accepted) or not a.x == c.x
    assert np.array_equal(np.asarray(start), before)


def test_basin_hopping_with_no_hops():
    result = BasinHopping(
        _objective, random_hop(), T=1.0, stepsize=0.5, rng=np.random.default_rng(0)
    ).run(_start(), 0)

    assert result.nit == 0
    for name in ("accepted", "stepsizes", "temperatures", "scores"):
        assert result[name].shape == (0,)
    assert np.asarray(result.poses).shape == (0, LAYOUT.n_dims)


def test_basin_hopping_adapts_the_stepsize_and_every_run_starts_afresh():
    seen = []

    def recording_hop(pose, rng, stepsize):
        seen.append(stepsize)
        return random_hop(BOUNDS)(pose, rng, stepsize)

    # Always accepting (hot) -> the rate is 1 > target -> the step size must grow.
    bh = BasinHopping(
        _objective,
        recording_hop,
        T=1e9,
        stepsize=0.5,
        adapt_stepsize=adaptive_stepsize(window=3, hi=3.0),
        rng=np.random.default_rng(0),
    )
    bh.run(_start(), 12)
    first = list(seen)
    seen.clear()
    bh.run(_start(), 12)

    assert first[0] == 0.5 and first[-1] > 0.5 and max(first) <= 3.0
    assert seen[0] == 0.5  # the second run does not inherit the first run's step size
    assert np.array_equal(first, seen)


def test_basin_hopping_calls_the_temperature_schedule_with_hop_and_total():
    calls = []

    def schedule(i, niter):
        calls.append((i, niter))
        return 1.0

    BasinHopping(
        _objective, random_hop(), T=schedule, stepsize=0.5, rng=np.random.default_rng(0)
    ).run(_start(), 7)

    assert calls == [(i, 7) for i in range(7)]


# ---------------------------------------------------------------------------
# boltzmann_diversity_filter
# ---------------------------------------------------------------------------


def _clustered_poses(n: int, layout: PoseLayout, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-10, 10, (6, 3))
    translation = centers[rng.integers(0, 6, n)] + rng.normal(scale=1.0, size=(n, 3))
    energies = rng.gamma(2.0, 3.0, n) - 10 + 0.7 * np.linalg.norm(translation, axis=1)
    poses = Poses.from_parts(
        layout.sample_random_rotations(n, rng),
        translation,
        layout.sample_random_torsions(n, rng),
    )
    return poses, energies


def _mean_nearest_neighbour(translation: np.ndarray) -> float:
    d = np.linalg.norm(translation[:, None] - translation[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(d.min(axis=1).mean())


def test_diversity_filter_returns_distinct_input_poses():
    poses, energies = _clustered_poses(600, LAYOUT)
    before = np.asarray(poses).copy()

    out = boltzmann_diversity_filter(poses, energies, 25, rng=np.random.default_rng(1))

    assert isinstance(out, Poses) and out.layout == LAYOUT and len(out) == 25
    rows = {tuple(r) for r in np.asarray(poses)}
    picked = [tuple(r) for r in np.asarray(out)]
    assert all(r in rows for r in picked) and len(set(picked)) == 25
    assert np.array_equal(np.asarray(poses), before)  # the input is untouched


def test_diversity_filter_is_reproducible_from_the_rng():
    poses, energies = _clustered_poses(300, LAYOUT)

    a = boltzmann_diversity_filter(poses, energies, 10, rng=np.random.default_rng(7))
    b = boltzmann_diversity_filter(poses, energies, 10, rng=np.random.default_rng(7))
    c = boltzmann_diversity_filter(poses, energies, 10, rng=np.random.default_rng(8))

    assert a == b and not a == c


def test_diversity_filter_k_larger_than_the_number_of_poses_returns_all():
    poses, energies = _clustered_poses(40, LAYOUT)

    out = boltzmann_diversity_filter(poses, energies, 1000, rng=np.random.default_rng(0))

    assert len(out) == 40


def test_diversity_filter_spreads_out_more_than_the_naive_choices():
    poses, energies = _clustered_poses(600, LAYOUT)
    seeds = range(15)

    filtered = np.mean(
        [
            _mean_nearest_neighbour(
                boltzmann_diversity_filter(
                    poses, energies, 25, rng=np.random.default_rng(s)
                ).translation
            )
            for s in seeds
        ]
    )
    random_subset = np.mean(
        [
            _mean_nearest_neighbour(
                poses.translation[np.random.default_rng(s).choice(600, 25, replace=False)]
            )
            for s in seeds
        ]
    )
    lowest_energy = _mean_nearest_neighbour(poses.translation[np.argsort(energies)[:25]])
    flat = np.mean(
        [
            _mean_nearest_neighbour(
                boltzmann_diversity_filter(
                    poses, energies, 25, T=1e9, rng=np.random.default_rng(s)
                ).translation
            )
            for s in seeds
        ]
    )

    assert filtered > random_subset and filtered > lowest_energy
    assert flat > filtered  # weighting by energy trades some spread for good poses


def test_diversity_filter_cold_temperature_caps_the_pool_and_starts_from_the_best():
    poses, energies = _clustered_poses(200, LAYOUT)
    temperature, k = 1e-3, 5
    weights = np.exp(-(energies - energies.min()) / temperature)

    out = boltzmann_diversity_filter(
        poses, energies, k, T=temperature, rng=np.random.default_rng(0)
    )

    # Poses whose Boltzmann weight underflows to zero cannot be drawn, so (documented) fewer than
    # k poses can come back; the first one picked is the best of the pool, which is the best overall.
    assert len(out) == min(k, np.count_nonzero(weights)) < k
    assert np.array_equal(np.asarray(out)[0], np.asarray(poses)[int(np.argmin(energies))])


def test_diversity_filter_selects_on_translation_for_every_layout():
    euler, energies = _clustered_poses(300, PoseLayout("euler", 3), seed=4)
    quat_layout = PoseLayout("quat", 3)
    quat = Poses.from_parts(
        quat_layout.sample_random_rotations(300, np.random.default_rng(9)),
        euler.translation,
        euler.torsions,
    )

    a = boltzmann_diversity_filter(euler, energies, 12, rng=np.random.default_rng(3))
    b = boltzmann_diversity_filter(quat, energies, 12, rng=np.random.default_rng(3))

    assert np.array_equal(a.translation, b.translation)


def test_diversity_filter_handles_equal_energies_and_validates_its_input():
    poses, energies = _clustered_poses(100, LAYOUT)

    assert len(boltzmann_diversity_filter(poses, np.full(100, 3.0), 10)) == 10
    for bad in ({"k": 0}, {"k": 5, "T": 0.0}, {"k": 5, "T": -1.0}):
        with pytest.raises(ValueError):
            boltzmann_diversity_filter(poses, energies, **bad)
    with pytest.raises(ValueError):
        boltzmann_diversity_filter(poses, energies[:-1], 5)
