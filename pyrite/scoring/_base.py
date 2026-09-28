from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from .dependencies import Dependency
from .. import Mol
import copy

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


    # TODO: should this take one mol? Multiple?
    def step(self, x: NDArray, mol: Mol) -> float:
        """The step function.

        This function updates the `mol` pose based on the supplied parameters `x` and returns
        the score associated with the update pose. The :class:`~pyrite.Mol` is modified using
        :meth:`~pyrite.Mol.update`, using a new conformer.
        The `mol` global conformer thus remains unchanged.

        Parameters
        ----------
        x : ndarray
            The variables describing the molecule pose. The shape should be
            ``(6 + n_dihedrals,)``, like ``[roll, pitch, yaw, x, y, z, *dihedrals]``.

        mol : Mol
            The molecule for which the pose will be evaluated.

        Returns
        -------
        score : float
            The score associated with the input pose.
        """
        conf_id = mol.update(x, new_conf=True)
        score = self.get_score(conf_id)
        mol.RemoveConformer(conf_id)
        return score


    def get_score(self, conf_id: int = -1, subscores: dict[ScoringFunction, float] | None = None) -> float:
        """Retrieves the score.

        This method first retrieves all dependencies of this ``ScoringFunction`` instance, merges
        them, and then resolves them. It then calls the ``_score`` function, which calculates
        the score using the computed dependencies.

        .. note::
            The dependency set and its merge are cached on the instance after the first call,
            since they only depend on the (fixed) composition of the scoring function, not on
            `conf_id`. Only ``Dependency.compute`` re-runs on every call.

        Parameters
        ----------
        conf_id : int, default -1
            The conformer id for which to calculate the score. Uses the global conformer by
            default.

        Returns
        -------
        score : float
            The score associated with the conformer.
        """

        opt_deps = getattr(self, "_opt_deps_cache", None)
        if opt_deps is None:
            raw_deps = self.get_dependencies()
            opt_deps = Dependency.merge_all(raw_deps)
            self._opt_deps_cache = opt_deps
        computed = {dep: dep.compute(conf_id) for dep in opt_deps}

        if subscores is not None:
            return self._score_and_store(conf_id, computed, subscores)
        return self._score(conf_id, computed=computed)

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

    def _score_and_store(self, conf_id: int, computed: dict[Dependency, Any], subscores: dict[ScoringFunction, float]):
        score = self._score(conf_id, computed=computed)
        subscores[self] = score
        return score

    @abstractmethod
    def _score(self, conf_id: int, computed: dict[Dependency, Any] | None) -> float:
        """The score function.

        This function takes the conformer id and the computed dependencies as input and returns
        the associated score.

        .. note::
            Do not call this method directly. Use ``get_score`` instead.

        :meta public:

        Parameters
        ----------
        conf_id : int
            The conformer id for which to calculate the score.
        computed : dict[Dependency, Any]
            A dictionary containing the computed dependencies. This is supplied by ``get_score``.

        Returns
        -------
        float
        """

    def get_dependencies(self) -> set[Dependency]:
        """Get the dependencies of this scoring function.

        This method returns a set of the :class:`~pyrite.scoring.dependencies.Dependency` that are
        used in this scoring function.

        When combining scoring functions, this method returns the set of all dependencies of all
        combined scoring functions.

        Returns
        -------
        set
        """
        return set()

    def __neg__(self):
        return _ScaledScoringFunction(-1, "*", self)

    def __add__(self, other):
        if isinstance(other, ScoringFunction):
            return _CombinedScoringFunction(self, other)
        if isinstance(other, (float, int)):
            return _CombinedScoringFunction(self, ConstantTerm(other))
        raise TypeError(
            f"Unsupported operand type(s) for +: 'ScoringFunction' and '{type(other)}'"
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, ScoringFunction):
            return _CombinedScoringFunction(self, -other)
        if isinstance(other, (float, int)):
            return _CombinedScoringFunction(self, ConstantTerm(-other))
        raise TypeError(
            f"Unsupported operand type(s) for -: 'ScoringFunction' and '{type(other)}'"
        )

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

    def get_dependencies(self) -> set[Dependency]:
        deps = set()
        for func in self.funcs:
            deps.update(func.get_dependencies())
        return deps

    def _score(self, conf_id, computed) -> float:
        total = 0.0
        func: ScoringFunction
        for func in self.funcs:
            # pylint: disable=protected-access
            total += func._score(conf_id, computed=computed)
        return total

    def _score_and_store(self, conf_id: int, computed: dict[Dependency, Any], subscores: dict[ScoringFunction, float]):
        total = 0.0
        func: ScoringFunction
        for func in self.funcs:
            # pylint: disable=protected-access
            total += func._score_and_store(conf_id, computed=computed, subscores=subscores)
        subscores[self] = total
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

    def get_dependencies(self) -> set[Dependency]:
        deps = set()
        if isinstance(self.left, ScoringFunction):
            deps.update(self.left.get_dependencies())
        if isinstance(self.right, ScoringFunction):
            deps.update(self.right.get_dependencies())
        return deps

    def _score(self, conf_id, computed) -> float:
        # pylint: disable=protected-access

        left_val = self.left
        right_val = self.right
        if isinstance(self.left, ScoringFunction):
            left_val = self.left._score(conf_id, computed=computed)
        if isinstance(self.right, ScoringFunction):
            right_val = self.right._score(conf_id, computed=computed)

        match self.operator:
            case "*":
                return left_val * right_val
            case "/":
                return left_val / right_val
            case "^":
                return left_val**right_val
            case _:
                raise TypeError(f"Unsupported operator for scaling: '{self.operator}'")

    def _score_and_store(self, conf_id, computed, subscores) -> float:
        # pylint: disable=protected-access

        left_val = self.left
        right_val = self.right
        if isinstance(self.left, ScoringFunction):
            left_val = self.left._score_and_store(conf_id, computed=computed, subscores=subscores)
        if isinstance(self.right, ScoringFunction):
            right_val = self.right._score_and_store(conf_id, computed=computed, subscores=subscores)

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

        if min_score <= max_score:
            raise ValueError("min_score must be <= max_score")

        self.min_score = min_score
        self.max_score = max_score

    def _score(self, *args, **kwargs) -> float:
        # pylint: disable=protected-access

        return np.clip(
            self.scoring_function._score(*args, **kwargs),
            self.min_score,
            self.max_score,
        )

    def _score_and_store(self, conf_id, computed, subscores) -> float:
        score = np.clip(
            self.scoring_function._score_and_store(conf_id, computed=computed, subscores=subscores),
            self.min_score,
            self.max_score,
        )
        subscores[self] = score
        return score

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

