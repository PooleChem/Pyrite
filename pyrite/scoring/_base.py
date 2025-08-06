from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from .dependencies import Dependency
from .. import Ligand, Receptor
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


class ScoringFunction(ABC):
    """
    Scoring Function base class.

    A :class:`ScoringFunction` is used to score pyrite poses.

    .. note::
        This is an abstract base class and should thus be subclassed. Please refer to the Notes
        section for more information on how to do this.


    Parameters
    ----------
    ligand : Ligand
        The ligand to be used in the ``ScoringFunction``.
    receptor : Receptor
        The receptor to be used in the ``ScoringFunction``.



    """

    def __init__(self, ligand: Ligand, receptor: Receptor):
        self.ligand = ligand
        self.receptor = receptor

        # self.computed = None

    def get_step(self, ligand: Ligand) -> Callable[[NDArray], float]:
        """Get a :meth:`step` function that does not need a :class:`~pyrite.Ligand` as input.

        This is useful if the optimizer used does not allow for additional step function arguments.

        Parameters
        ----------
        ligand : Ligand
            The ``Ligand`` instance to be used in the step function.

        Returns
        -------
        step_function : callable
        """

        def step(x):
            conf_id = ligand.update(x, new_conf=True)
            score = self.get_score(conf_id)
            ligand.RemoveConformer(conf_id)

            return score

        return step

    def step(self, x: NDArray, ligand: Ligand) -> float:
        """The step function.

        This function updates the `ligand` pose based on the supplied parameters `x` and returns
        the score associated with the update pose. The :class:`~pyrite.Ligand` is modified using
        :meth:`~pyrite.Ligand.update`, using a new conformer.
        The `ligand` global conformer thus remains unchanged.

        Parameters
        ----------
        x : ndarray
            The variables describing the ligand pose. The shape should be
            ``(6 + n_dihedrals,)``, like ``[roll, pitch, yaw, x, y, z, *dihedrals]``.

        ligand : Ligand
            The ligand for which the pose will be evaluated.

        Returns
        -------
        score : float
            The score associated with the input pose.
        """
        conf_id = ligand.update(x, new_conf=True)
        score = self.get_score(conf_id)
        ligand.RemoveConformer(conf_id)
        return score

    def get_score(self, conf_id: int = -1) -> float:
        """Retrieves the score.

        This method first retrieves all dependencies of this ``ScoringFunction`` instance, merges
        them, and then resolves them. It then calls the ``_score`` function, which calculates
        the score using the computed dependencies.

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

        raw_deps = self.get_dependencies()
        opt_deps = Dependency.merge_all(raw_deps)
        computed = {dep: dep.compute(conf_id) for dep in opt_deps}

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


class MyProblem:
    def __init__(
        self,
        ligand2,
        binding_site2,
        scoring_function: ScoringFunction,
        dihedral_quantize_degree: int = 0,
    ):
        self.ligand = ligand2
        self.binding_site = binding_site2
        self.scoring_function = scoring_function
        self.dihedral_quantize_degree = dihedral_quantize_degree

        bounds2 = [(-2 * np.pi, 2 * np.pi)] * (6 + len(self.ligand.dihedral_angles))
        bounds2[3:6] = self.binding_site.get_translation_bounds()

        if self.dihedral_quantize_degree > 0:
            self._bins = int(360 // self.dihedral_quantize_degree)
            if 360 % self._bins != 0:
                raise ValueError(
                    "dihedral_quantize_degree must be a divisor of 360 degrees"
                )
            bounds2[6:] = [(-self._bins, self._bins)] * len(self.ligand.dihedral_angles)

        self.bounds = ([x[0] for x in bounds2], [x[1] for x in bounds2])

    def quantize_dihedrals(self, vs):
        v_2d = np.atleast_2d(vs)

        b_div = (2 * np.pi) / self._bins
        v_2d[:, 6:] //= b_div

        if vs.ndim < 2:
            return v_2d[0, :]

        return v_2d

    def dequantize_dihedrals(self, vs):
        v_2d = np.atleast_2d(vs)

        b_div = (2 * np.pi) / self._bins
        v_2d[:, 6:] *= b_div

        if vs.ndim < 2:
            return v_2d[0, :]
        return v_2d

    def get_nix(self):
        if self.dihedral_quantize_degree > 0:
            return len(self.ligand.dihedral_angles)
        return 0

    def __deepcopy__(self, memo):
        # create blank instance
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        # shallow‐copy ligand to preserve its subclass
        new.ligand = self.ligand
        new.scoring_function = self.scoring_function
        new.binding_site = self.binding_site
        # deep‐copy everything else
        for k, v in self.__dict__.items():
            if k != "ligand" and k != "scoring_function" and k != "binding_site":
                setattr(new, k, copy.deepcopy(v, memo))
        return new

    def get_bounds(self):
        return self.bounds

    def fitness(self, x):
        if self.dihedral_quantize_degree > 0:
            deq_x = self.dequantize_dihedrals(x)
        else:
            deq_x = x

        return [self.scoring_function.step(deq_x, self.ligand)]
