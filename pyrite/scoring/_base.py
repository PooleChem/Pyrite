from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._common import Mol, Pose, Poses
from .dependencies import Dependency, RDKitDependency, Realization, _NarrowingComputed

# speed grade:
"""
<= 100 : 🐌
<= 1000 : 🐢
<= 6000 : 🚶
<= 12000 : 🚲
<= 30000 : 🚗
<= 60000 : 🚄
<= 100000 : ✈️
>= 100000 : 🚀
"""
# TODO: make more processor independent (relative to ligand transform?)


class ScoringFunction(ABC):
    """
    Scoring Function base class.

    A :class:`ScoringFunction` is used to score pyrite poses.

    .. note::
        This is an abstract base class and should thus be subclassed. Please refer to the Notes
        section for more information on how to do this.

    """

    def __init__(
        self,
    ):
        pass

    def _resolved_dependencies(self) -> list[Dependency]:
        """The merged dependencies. Cached: this only depends on the (fixed) composition of the
        scoring function.
        """
        opt_deps = getattr(self, "_opt_deps_cache", None)
        if opt_deps is None:
            opt_deps = list(Dependency.merge_all(self.get_dependencies()))
            self._opt_deps_cache = opt_deps
        return opt_deps

    def get_score(
        self, pose: Pose | NDArray, subscores: dict[ScoringFunction, float] | None = None
    ) -> float:
        """Retrieves the score of a pose.

        This method first retrieves all dependencies of this ``ScoringFunction`` instance, merges
        them, and then resolves them for the pose. It then calls the ``_score`` function, which
        calculates the score using the computed dependencies.

        Nothing is done to the :class:`~pyrite.Mol`: the atom positions are computed from the pose
        directly, and a private RDKit copy is made only if a term needs one, so poses of one
        molecule can be scored from several threads at once.

        .. note::
            The dependency set and its merge are cached on the instance after the first call,
            since they only depend on the (fixed) composition of the scoring function, not on
            `pose`. Only ``Dependency.compute`` re-runs on every call.

        Parameters
        ----------
        pose : Pose, ndarray
            The pose to score, or its raw values, like ``[roll, pitch, yaw, x, y, z, *torsions]``.
        subscores : dict[ScoringFunction, float], optional
            If given, filled with the score of every part of a composite scoring function.

        Returns
        -------
        score : float
            The score associated with the pose.
        """
        realized = Realization(pose, batched=False)
        # _NarrowingComputed, not a plain dict: opt_deps holds one representative
        # per merged group, but different dependencies sharing that group (e.g.
        # different k/cutoff) still need their own narrowed view back — see
        # Dependency.narrow()'s docstring for why that can't be a plain dict.
        computed = _NarrowingComputed(
            {dep: dep.compute(realized) for dep in self._resolved_dependencies()}
        )

        if subscores is not None:
            return self._score_and_store(pose, computed, subscores)
        return self._score(pose, computed=computed)

    def batch_scores(self, poses: Poses | NDArray) -> NDArray[np.float64]:
        """Score many poses at once.

        Mirrors ``get_score`` — resolves and merges dependencies once for the whole
        call, then calls ``_batch_scores``. The default implementation is just a loop
        over ``get_score`` (correct for every scoring function, no speedup); subclasses
        that can do better override ``_batch_scores``, not this method.

        Parameters
        ----------
        poses : Poses, ndarray
            The poses to score, or their raw values, shape ``(n_poses, n_dims)``.

        Returns
        -------
        NDArray
            One score per pose, same order as `poses`.
        """
        realized = Realization(poses, batched=True)
        computed_batch = _NarrowingComputed(
            {dep: dep.compute(realized) for dep in self._resolved_dependencies()}
        )
        return self._batch_scores(poses, computed_batch)

    def _batch_scores(self, poses, computed_batch) -> NDArray[np.float64]:
        """The batched score function.

        .. note::
            Do not call this method directly. Use ``batch_scores`` instead.

        :meta public:

        Parameters
        ----------
        poses : Poses, ndarray
            The poses to score.
        computed_batch : dict[Dependency, Any]
            The batched equivalent of ``_score``'s `computed` — supplied by
            ``batch_scores``, with a leading ``n_poses`` axis on everything. The default
            implementation calls ``_score`` for every pose on its own entry of it
            (``computed_batch.row(i)``), so the dependencies are still computed once for the
            whole batch; subclasses override this method to do better than one-at-a-time.

        Returns
        -------
        NDArray
        """
        return np.array([self._score(pose, computed_batch.row(i)) for i, pose in enumerate(poses)])

    def clamp(
        self,
        min_score: float = -float("inf"),
        max_score: float = float("inf"),
    ):
        """Return a clamped version of this scoring function.

        Any score values below `min_score` will be set to `min_score`, and any values above
        `max_score` will be set to `max_score`.

        Parameters
        ----------
        min_score : float, default -inf
            The minimum score to clamp to. Any values below this will be set to this.
        max_score : float, default inf
            The maximum score to clamp to. Any values above this will be set to this.

        Returns
        -------
        Clamp
        """
        return Clamp(self, min_score, max_score)

    def _score_and_store(
        self,
        pose: Pose,
        computed: dict[Dependency, Any],
        subscores: dict[ScoringFunction, float],
    ):
        score = self._score(pose, computed=computed)
        subscores[self] = score
        return score

    @abstractmethod
    def _score(self, pose: Pose, computed: dict[Dependency, Any] | None) -> float:
        """The score function.

        This function takes the pose and the computed dependencies as input and returns
        the associated score.

        .. note::
            Do not call this method directly. Use ``get_score`` instead.

        :meta public:

        Parameters
        ----------
        pose : Pose, ndarray
            The pose for which to calculate the score.
        computed : dict[Dependency, Any]
            A dictionary containing the computed dependencies. This is supplied by ``get_score``.

        Returns
        -------
        float
        """

    def get_dependencies(self) -> list[Dependency]:
        """Get the dependencies of this scoring function.

        This method returns a list of the :class:`~pyrite.scoring.dependencies.Dependency` that are
        used in this scoring function.

        When combining scoring functions, this method returns every dependency of every combined
        scoring function — deliberately a ``list``, not a ``set``: two dependencies can compare
        equal (same ``group_key``, e.g. same point cloud/query — enough to be merged into one
        shared computation) while still being different instances with different parameters (e.g.
        ``k``/cutoff) that each need their own narrowed view back (see
        :meth:`~pyrite.scoring.dependencies.Dependency.narrow`). A ``set`` would silently collapse
        those down to one arbitrary survivor before :meth:`~pyrite.scoring.dependencies.Dependency.merge_all`
        ever saw the others. ``merge_all`` does its own grouping/deduplication downstream and works
        fine with a list that has such "duplicates" in it.

        Returns
        -------
        list
        """
        return []

    def __neg__(self):
        return _ScaledScoringFunction(-1, "*", self)

    def __add__(self, other):
        if isinstance(other, ScoringFunction):
            return _CombinedScoringFunction(self, other)
        if isinstance(other, (float, int)):
            return _CombinedScoringFunction(self, ConstantTerm(other))
        raise TypeError(f"Unsupported operand type(s) for +: 'ScoringFunction' and '{type(other)}'")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, ScoringFunction):
            return _CombinedScoringFunction(self, -other)
        if isinstance(other, (float, int)):
            return _CombinedScoringFunction(self, ConstantTerm(-other))
        raise TypeError(f"Unsupported operand type(s) for -: 'ScoringFunction' and '{type(other)}'")

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __mul__(self, other):
        if not isinstance(other, (int, float, ScoringFunction)):
            raise TypeError(
                f"Unsupported operand type(s) for *: 'ScoringFunction' and '{type(other)}'"
            )
        return _ScaledScoringFunction(other, "*", self)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if not isinstance(other, (int, float, ScoringFunction)):
            raise TypeError(
                f"Unsupported operand type(s) for /: 'ScoringFunction' and '{type(other)}'"
            )
        return _ScaledScoringFunction(self, "/", other)

    def __rtruediv__(self, other):
        if not isinstance(other, (int, float, ScoringFunction)):
            raise TypeError(
                f"Unsupported operand type(s) for /: '{type(other)}' and 'ScoringFunction'"
            )
        return _ScaledScoringFunction(other, "/", self)

    def __pow__(self, other):
        if not isinstance(other, (int, float, ScoringFunction)):
            raise TypeError(
                f"Unsupported operand type(s) for **: 'ScoringFunction' and '{type(other)}'"
            )
        return _ScaledScoringFunction(self, "^", other)

    def __rpow__(self, other):
        if not isinstance(other, (int, float, ScoringFunction)):
            raise TypeError(
                f"Unsupported operand type(s) for **: '{type(other)}' and 'ScoringFunction'"
            )
        return _ScaledScoringFunction(other, "^", self)

    def __repr__(self):
        return type(self).__name__


class _CombinedScoringFunction(ScoringFunction):  # pylint: disable=too-few-public-methods
    """
    Sums the score of multiple scoring functions.

    .. note::
        This is an internal class. Use ``+`` instead.


    Parameters
    ----------
    *functions : ScoringFunction
        All arguments are considered as a scoring function to sum.

    Attributes
    ----------
    funcs : list[ScoringFunction]
        The list of scoring functions.

    """

    def __init__(self, *functions: ScoringFunction):
        super().__init__()

        self.funcs = []
        for func in functions:
            if isinstance(func, _CombinedScoringFunction):
                self.funcs.extend(func.funcs)
            else:
                self.funcs.append(func)

    def get_dependencies(self) -> list[Dependency]:
        deps = []
        for func in self.funcs:
            deps.extend(func.get_dependencies())
        return deps

    def _score(self, pose, computed) -> float:
        total = 0.0
        func: ScoringFunction
        for func in self.funcs:
            # pylint: disable=protected-access
            total += func._score(pose, computed=computed)
        return total

    def _score_and_store(
        self,
        pose: Pose,
        computed: dict[Dependency, Any],
        subscores: dict[ScoringFunction, float],
    ):
        total = 0.0
        func: ScoringFunction
        for func in self.funcs:
            # pylint: disable=protected-access
            total += func._score_and_store(pose, computed=computed, subscores=subscores)
        subscores[self] = total
        return total

    def _score_field(self, r, idx, atom_type):
        # pylint: disable=protected-access
        total = np.zeros(len(r))
        for func in self.funcs:
            total = total + func._score_field(r, idx, atom_type)
        return total

    def _batch_scores(self, poses, computed_batch) -> NDArray[np.float64]:
        # pylint: disable=protected-access
        total = np.zeros(len(poses))
        for func in self.funcs:
            total = total + func._batch_scores(poses, computed_batch)
        return total

    def __repr__(self):
        return f"<{type(self).__name__}: {' + '.join(map(str, self.funcs))}>"


class _ScaledScoringFunction(ScoringFunction):
    """
    Scales scoring functions.

    Supports two-way multiplication (``*``), division (``/``), and exponentiation (``**``).

    .. note::
        This is an internal class. Use ``*``, ``/`` or ``**`` instead.


    Parameters
    ----------
    left : ScoringFunction
        The ``ScoringFunction`` to the left side of the operator.
    operator : {'*', '/', '^'}
        The operator to use.
    right : ScoringFunction
        The ``ScoringFunction`` to the right side of the operator.

    Attributes
    ----------
    left : ScoringFunction, float
        The ``ScoringFunction`` to the left side of the operator.
    operator : {'*', '/', '^'}
        The operator to use.
    right : ScoringFunction, float
        The ``ScoringFunction`` to the right side of the operator.
    """

    def __init__(
        self,
        left: ScoringFunction | float,
        operator: str,
        right: ScoringFunction | float,
    ):
        self.left = left
        self.operator = operator
        self.right = right

    def get_dependencies(self) -> list[Dependency]:
        deps = []
        if isinstance(self.left, ScoringFunction):
            deps.extend(self.left.get_dependencies())
        if isinstance(self.right, ScoringFunction):
            deps.extend(self.right.get_dependencies())
        return deps

    def _score(self, pose, computed) -> float:
        # pylint: disable=protected-access

        left_val = self.left
        right_val = self.right
        if isinstance(self.left, ScoringFunction):
            left_val = self.left._score(pose, computed=computed)
        if isinstance(self.right, ScoringFunction):
            right_val = self.right._score(pose, computed=computed)

        match self.operator:
            case "*":
                return left_val * right_val
            case "/":
                return left_val / right_val
            case "^":
                return left_val**right_val
            case _:
                raise TypeError(f"Unsupported operator for scaling: '{self.operator}'")

    def _score_and_store(self, pose, computed, subscores) -> float:
        # pylint: disable=protected-access

        left_val = self.left
        right_val = self.right
        if isinstance(self.left, ScoringFunction):
            left_val = self.left._score_and_store(pose, computed=computed, subscores=subscores)
        if isinstance(self.right, ScoringFunction):
            right_val = self.right._score_and_store(pose, computed=computed, subscores=subscores)

        score = 0
        match self.operator:
            case "*":
                score = left_val * right_val
            case "/":
                score = left_val / right_val
            case "^":
                score = left_val**right_val
            case _:
                raise TypeError(f"Unsupported operator for scaling: '{self.operator}'")

        subscores[self] = score
        return score

    def _score_field(self, r, idx, atom_type):
        # pylint: disable=protected-access

        left_val = self.left
        right_val = self.right
        if isinstance(self.left, ScoringFunction):
            left_val = self.left._score_field(r, idx, atom_type)
        if isinstance(self.right, ScoringFunction):
            right_val = self.right._score_field(r, idx, atom_type)

        match self.operator:
            case "*":
                return left_val * right_val
            case "/":
                return left_val / right_val
            case "^":
                return left_val**right_val
            case _:
                raise TypeError(f"Unsupported operator for scaling: '{self.operator}'")

    def _batch_scores(self, poses, computed_batch) -> NDArray[np.float64]:
        # pylint: disable=protected-access

        left_val = self.left
        right_val = self.right
        if isinstance(self.left, ScoringFunction):
            left_val = self.left._batch_scores(poses, computed_batch)
        if isinstance(self.right, ScoringFunction):
            right_val = self.right._batch_scores(poses, computed_batch)

        match self.operator:
            case "*":
                return left_val * right_val
            case "/":
                return left_val / right_val
            case "^":
                return left_val**right_val
            case _:
                raise TypeError(f"Unsupported operator for scaling: '{self.operator}'")

    def __str__(self):
        return f"{self.left} {self.operator} {self.right}"

    def __repr__(self):
        return f"<{type(self).__name__}: {self.left.__repr__()} {self.operator} {self.right.__repr__()}>"


class Clamp(ScoringFunction):
    """
    Clamps the output of a scoring function.

    Parameters
    ----------
    scoring_function : ScoringFunction
        The scoring function to clamp.
    min_score : float, default -inf
        The minimum score to clamp to. Any values below this will be set to this.
    max_score : float, default inf
        The maximum score to clamp to. Any values above this will be set to this.


    Raises
    ------
    ValueError
        If `min_score` is greater than `max_score`.

    """

    def __init__(
        self,
        scoring_function: ScoringFunction,
        min_score: float = -float("inf"),
        max_score: float = float("inf"),
    ):
        self.scoring_function = scoring_function

        if min_score > max_score:
            raise ValueError("min_score must be <= max_score")

        self.min_score = min_score
        self.max_score = max_score

    def get_dependencies(self) -> list[Dependency]:
        return self.scoring_function.get_dependencies()

    def _score(self, *args, **kwargs) -> float:
        # pylint: disable=protected-access

        return np.clip(
            self.scoring_function._score(*args, **kwargs),
            self.min_score,
            self.max_score,
        )

    def _score_and_store(self, pose, computed, subscores) -> float:
        score = np.clip(
            self.scoring_function._score_and_store(pose, computed=computed, subscores=subscores),
            self.min_score,
            self.max_score,
        )
        subscores[self] = score
        return score

    def _score_field(self, r, idx, atom_type):
        # pylint: disable=protected-access
        return np.clip(
            self.scoring_function._score_field(r, idx, atom_type),
            self.min_score,
            self.max_score,
        )

    def _batch_scores(self, poses, computed_batch) -> NDArray[np.float64]:
        # pylint: disable=protected-access
        return np.clip(
            self.scoring_function._batch_scores(poses, computed_batch),
            self.min_score,
            self.max_score,
        )

    def __repr__(self):
        return f"<{type(self).__name__}: {self.scoring_function.__repr__()} in [{self.min_score}, {self.max_score}]>"


class ConstantTerm(ScoringFunction):
    """
    Represents a constant scoring term.

    This class is used whenever a ``ScoringFunction`` is summed with a constant value. It can also
    be used to indicate a constant term explicitly.


    Parameters
    ----------
    constant : float
        The value of the constant term.

    """

    def __init__(self, constant: float):
        self.constant = constant

    def _score(self, *args, **kwargs) -> float:
        return self.constant

    def _score_field(self, r, idx, atom_type):
        return np.full(len(r), self.constant)

    def _batch_scores(self, poses, computed_batch) -> NDArray[np.float64]:
        return np.full(len(poses), self.constant)


class _RDKitScoringFunction(ScoringFunction, ABC):
    """Base class for scoring functions that need a real RDKit conformer of the molecule.

    Implement ``_score(pose, computed)`` as for any scoring function, and read the posed molecule
    with ``computed[self.rdkit_dep]``: an :class:`~rdkit.Chem.rdchem.Mol` copy of ``self.mol`` with
    the pose as its only conformer (``confId=-1``), which is private to this call and may be used
    freely. It is made by a shared :class:`~pyrite.scoring.dependencies.RDKitDependency`, so any
    number of these terms in one composite share one copy per pose, and ``self.mol`` is never
    modified.

    Parameters
    ----------
    mol : Mol
        The molecule to make a conformer of.
    """

    def __init__(self, mol: Mol):
        self.mol = mol
        self.rdkit_dep = RDKitDependency(mol)

    def get_dependencies(self) -> list[Dependency]:
        return [self.rdkit_dep]
