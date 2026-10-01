from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize

from pyrite._common import Pose, Poses


def random_hop(
    translation_bounds: NDArray | None = None,
) -> Callable[[Pose, np.random.Generator, float], Pose]:
    """Create the standard random hop for :class:`BasinHopping`.

    The rotation is perturbed on SO(3) by a Gaussian rotation vector with standard deviation
    `stepsize`, composed on the left of the current rotation (see
    :meth:`~pyrite._common.PoseLayout.compose_rotation`), so the step does not depend on the
    current orientation. Translation and torsions each get independent uniform noise in
    ``[-stepsize, stepsize]``. Torsions are not wrapped, since the score is periodic in them.

    The `stepsize` (radians for rotation and torsions, Angstrom for translation) is supplied on
    every call by :class:`BasinHopping`, which owns it so that it can be adapted.

    Parameters
    ----------
    translation_bounds : array_like, optional
        Shape ``(3, 2)`` of ``(lo, hi)`` per axis, as returned by
        :meth:`~pyrite.bounds.Bounds.get_translation_bounds`. Hopped translations are clipped to it.

    Returns
    -------
    hop : callable
        ``hop(pose, rng, stepsize) -> pose_new``. Returns a new pose, `pose` is not modified.
    """
    bounds = None if translation_bounds is None else np.asarray(translation_bounds, dtype=float)

    def hop(pose: Pose, rng: np.random.Generator, stepsize: float) -> Pose:
        layout = pose.layout
        rot, trans, tors = layout.rot_slice, layout.trans_slice, layout.tors_slice
        out = np.asarray(pose).astype(float)
        out[rot] = layout.compose_rotation(out[rot], rng.normal(size=3) * stepsize)
        out[trans] += rng.uniform(-stepsize, stepsize, size=3)
        if bounds is not None:
            out[trans] = np.clip(out[trans], bounds[:, 0], bounds[:, 1])
        out[tors] += rng.uniform(-stepsize, stepsize, size=out[tors].shape)
        return Pose(out, layout)

    return hop


def geometric_annealing(T_start: float, T_end: float) -> Callable[[int, int], float]:
    """Create a geometric temperature schedule for :class:`BasinHopping`.

    The temperature goes from `T_start` at the first hop to `T_end` at the last, decreasing by
    a constant factor per hop.

    Parameters
    ----------
    T_start, T_end : float
        The temperature of the first and last hop.

    Returns
    -------
    T : callable
        ``T(i, niter) -> float``, the temperature of hop `i` (0-based) in a run of `niter` hops.
    """

    def T(i: int, niter: int) -> float:
        ratio = (T_end / T_start) ** (1.0 / max(niter - 1, 1))
        return T_start * (ratio**i)

    return T


def adaptive_stepsize(
    target_rate: float = 0.4,
    window: int = 20,
    lo: float = 0.05,
    hi: float = 3.0,
    tolerance: float = 0.1,
    increase: float = 1.25,
    decrease: float = 0.8,
) -> Callable[[float, NDArray, int], float]:
    """Create an acceptance-rate feedback rule for the step size of :class:`BasinHopping`.

    Every `window` hops, the acceptance rate over the last `window` hops is compared with
    `target_rate`. If it is more than `tolerance` above, the step size is multiplied by
    `increase` (hops are too timid); if more than `tolerance` below, by `decrease` (hops are
    too bold). The step size is kept within ``[lo, hi]``.

    Parameters
    ----------
    target_rate : float, default 0.4
        The acceptance rate to aim for.
    window : int, default 20
        The number of hops between adjustments, and the number of hops the rate is measured over.
    lo, hi : float
        The bounds on the step size.
    tolerance : float, default 0.1
        The dead band around `target_rate` in which the step size is left alone.
    increase, decrease : float
        The factors applied to the step size.

    Returns
    -------
    adapt : callable
        ``adapt(stepsize, accepted, i) -> stepsize``, where `accepted` is the boolean acceptance
        history of the hops before hop `i`. It is a pure function; the state lives in
        :meth:`BasinHopping.run`, so one rule can be shared between runs.
    """

    def adapt(stepsize: float, accepted: NDArray, i: int) -> float:
        if i == 0 or i % window != 0 or len(accepted) < window:
            return stepsize
        rate = np.mean(accepted[-window:])
        if rate > target_rate + tolerance:
            return min(stepsize * increase, hi)
        if rate < target_rate - tolerance:
            return max(stepsize * decrease, lo)
        return stepsize

    return adapt


class BasinHopping:
    """Basin hopping global optimization.

    From the current minimum, hop to a nearby pose with `hop`, minimize locally from
    there, and accept the new minimum with the Metropolis criterion on the minimized scores.

    .. note::
        This is a global *optimizer*, not an MCMC sampler. The Hastings correction for the
        basin-to-basin proposal is intractable and is not applied, and `T` is not necessarily
        constant. The minima it visits are not Boltzmann samples of anything.

    Configure once, then call :meth:`run` for each starting pose.

    Parameters
    ----------
    func : callable
        The objective, ``func(pose) -> float``, taking a :class:`~pyrite._common.Pose`. Lower
        is better. For example ``lambda pose: scoring_function.step(pose, ligand)``.
    hop : callable
        The hop, ``hop(pose, rng, stepsize) -> pose_new``. It must return a new pose rather than
        modifying `pose`, and should draw all its randomness from `rng`. A hop without a notion
        of scale can ignore `stepsize`. See :func:`random_hop`.
    T : float or callable
        The Metropolis temperature, in the units of `func`. Either a constant, or a schedule
        ``T(i, niter) -> float`` (see :func:`geometric_annealing`). Floored at ``1e-6``.
    stepsize : float
        The initial step size passed to `hop`.
    adapt_stepsize : callable, optional
        A rule ``adapt(stepsize, accepted, i) -> stepsize`` called before every hop to update
        the step size from the acceptance history (see :func:`adaptive_stepsize`). By default
        the step size stays fixed. Every :meth:`run` starts again from `stepsize`.
    minimizer_kwargs : dict, optional
        Keyword arguments for :func:`scipy.optimize.minimize`. Defaults to L-BFGS-B with
        ``eps=1e-2`` and ``tol=1e-4``.
    rng : numpy.random.Generator, optional
        The single source of randomness, passed to `hop` and used for acceptance, so one
        seed reproduces a whole run. Defaults to a fresh generator (which, unlike the global
        ``np.random`` state, is not duplicated across forked worker processes).

    Examples
    --------
    >>> hop = random_hop(binding_site.get_translation_bounds())
    >>> bh = BasinHopping(
    ...     lambda pose: scoring.step(pose, ligand),
    ...     hop,
    ...     T=geometric_annealing(2.0, 0.1),
    ...     stepsize=0.5,
    ...     adapt_stepsize=adaptive_stepsize(),
    ... )
    >>> results = [bh.run(pose, niter=50) for pose in poses]
    """

    def __init__(
        self,
        func: Callable[[Pose], float],
        hop: Callable[[Pose, np.random.Generator, float], Pose],
        T: float | Callable[[int, int], float],
        stepsize: float,
        adapt_stepsize: Callable[[float, NDArray, int], float] | None = None,
        minimizer_kwargs: dict | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.func = func
        self.hop = hop
        self.T = T
        self.stepsize = stepsize
        self.adapt_stepsize = adapt_stepsize
        self.minimizer_kwargs = (
            {"method": "L-BFGS-B", "options": {"eps": 1e-2}, "tol": 1e-4}
            if minimizer_kwargs is None
            else dict(minimizer_kwargs)
        )
        self.rng = np.random.default_rng() if rng is None else rng

    def run(self, x0: Pose, niter: int) -> OptimizeResult[str, Any]:
        """Run `niter` hops from `x0`.

        Parameters
        ----------
        x0 : Pose
            The starting pose. It is minimized before the first hop, and its layout is used
            for every pose in the run.
        niter : int
            The number of hops.

        Returns
        -------
        scipy.optimize.OptimizeResult
            With ``x`` (a :class:`~pyrite._common.Pose`) and ``fun`` of the best minimum found,
            ``nit`` (the number of hops), and one entry per hop, each an array of length
            `niter`, so that the run can be inspected afterwards:

            - ``accepted``: whether the hop's minimum was accepted.
            - ``stepsizes``: the step size the hop was made with, after adaptation.
            - ``temperatures``: the temperature used in the hop's acceptance test.
            - ``scores``: the score of the minimum reached from the hop, whether or not it was
              accepted. (The score of the starting minimum is not included.)
            - ``poses``: the minimum reached from the hop, as a :class:`~pyrite._common.Poses`
              in the layout of `x0`, row for row with ``scores``.
        """
        layout = x0.layout

        def objective(v: NDArray) -> float:
            return self.func(Pose(v, layout))

        def minimize_from(pose: Pose) -> tuple[Pose, float]:
            res = minimize(objective, np.asarray(pose), **self.minimizer_kwargs)
            return Pose(res.x.copy(), layout), res.fun

        x, fx = minimize_from(x0)
        best_x, best_fx = x, fx
        accepted = np.zeros(niter, dtype=bool)
        stepsizes = np.zeros(niter)
        temperatures = np.zeros(niter)
        scores = np.zeros(niter)
        poses = np.zeros((niter, layout.n_dims))
        stepsize = self.stepsize

        for i in range(niter):
            if self.adapt_stepsize is not None:
                stepsize = self.adapt_stepsize(stepsize, accepted[:i], i)
            x_new, fx_new = minimize_from(self.hop(x, self.rng, stepsize))
            dE = fx_new - fx
            T = self.T(i, niter) if callable(self.T) else self.T
            stepsizes[i], temperatures[i] = stepsize, T
            scores[i], poses[i] = fx_new, np.asarray(x_new)
            accepted[i] = dE < 0 or self.rng.random() < np.exp(-dE / max(T, 1e-6))
            if accepted[i]:
                x, fx = x_new, fx_new
            if fx_new < best_fx:
                best_x, best_fx = x_new, fx_new

        return OptimizeResult(
            x=best_x,
            fun=best_fx,
            nit=niter,
            accepted=accepted,
            stepsizes=stepsizes,
            temperatures=temperatures,
            scores=scores,
            poses=Poses(poses, layout),
        )
