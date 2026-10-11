from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._common import Mol, Pose, Poses
from .dependencies import Dependency, RDKitDependency, Realization, _NarrowingComputed

# The speed grades in the docstrings (🚀 ✈️ 🚗 🚲 🐢 🐌) are measured with benchmarks/speed_grades.py;
# the legend is in the docstring of pyrite/scoring/__init__.py.


class ScoringFunction(ABC):
    """
    Scoring Function base class.

    A :class:`ScoringFunction` is used to score pyrite poses.

    .. note::
        This is an abstract base class and should thus be subclassed. Please refer to the Notes
        section for more information on how to do this.

    See Also
    --------
    pyrite.Mol : The molecule whose poses are scored.
    pyrite.scoring.grid.GridScore : A fast approximation of KNN-based scoring functions.
    pyrite.search.BasinHopping : Search for the pose with the lowest score.

    Notes
    -----
    A scoring function implements ``_score(pose, computed)``, which returns the score of one pose
    as a float. The simplest one computes everything itself, from the pose, with
    ``self.mol.pose_to_positions(pose)`` or ``self.mol.to_rdkit(pose)``. Everything else is
    optional:

    * ``get_dependencies()`` lists :class:`~pyrite.scoring.dependencies.Dependency` objects whose
      results ``_score`` reads from `computed`. Use these to share an expensive computation
      between terms, such as one nearest neighbor search for all terms on the same receptor.
    * ``_batch_scores(poses, computed_batch)`` scores many poses at once. The default calls
      ``_score`` for every pose.
    * ``_score_and_gradient(pose, computed)`` returns the score and its gradient with respect to
      the pose. The default uses finite differences. With the gradient per atom position, use
      :meth:`Mol.pose_gradient <pyrite.Mol.pose_gradient>`.

    The conformance tests in ``tests/test_contracts.py`` check a new scoring function
    automatically once it is registered there, including its gradient. The user guide page
    :doc:`/user_guide/writing_scoring_functions` walks through all of this with examples.

    Examples
    --------
    Scoring functions are combined with arithmetic, into one scoring function that shares its
    work (one nearest neighbor search for all terms on the same receptor):

    >>> from pyrite import Mol
    >>> ligand = Mol.from_sdf("ligand.sdf", flexible=True)
    >>> receptor = Mol.from_pdb("receptor.pdb")
    >>> from pyrite.scoring import Gaussian, Hydrophobic, NonDirHBond, NumTors, Repulsion
    >>> vina_like = (
    ...     -0.035579 * Gaussian(ligand, receptor, offset=0.0, width=0.5)
    ...     - 0.005156 * Gaussian(ligand, receptor, offset=3.0, width=2.0)
    ...     + 0.840245 * Repulsion(ligand, receptor)
    ...     - 0.035069 * Hydrophobic(ligand, receptor)
    ...     - 0.587439 * NonDirHBond(ligand, receptor)
    ... ) / (1 + 0.0585 * NumTors(ligand))
    >>> vina_like.get_score(ligand.input_pose)

    A new scoring function implements ``_score``:

    >>> import numpy as np
    >>> from pyrite.scoring import ScoringFunction
    >>> class RadiusOfGyration(ScoringFunction):
    ...     def __init__(self, mol):
    ...         self.mol = mol
    ...
    ...     def _score(self, pose, computed):
    ...         positions = self.mol.pose_to_positions(pose)
    ...         return float(np.sqrt(((positions - positions.mean(axis=0)) ** 2).sum(axis=1).mean()))
    """

    def __init__(
        self,
    ):
        pass

    def _resolved_dependencies(self) -> list[Dependency]:
        """Return the merged dependencies of this scoring function.

        Cached: this only depends on the (fixed) composition of the scoring function.
        """
        opt_deps = getattr(self, "_opt_deps_cache", None)
        if opt_deps is None:
            opt_deps = list(Dependency.merge_all(self.get_dependencies()))
            self._opt_deps_cache = opt_deps
        return opt_deps

    def get_score(
        self, pose: Pose | NDArray, subscores: dict[ScoringFunction, float] | None = None
    ) -> float:
        """Retrieve the score of a pose.

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
        float
            The score associated with the pose.

        See Also
        --------
        batch_scores : Score many poses at once.
        get_score_and_gradient : The score and its gradient.

        Examples
        --------
        >>> score = scoring_function.get_score(ligand.input_pose)
        >>> subscores = {}
        >>> score = scoring_function.get_score(pose, subscores=subscores)  # the score of every term
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
            score = self._score_and_store(pose, computed, subscores)
            subscores.update({term: float(value) for term, value in subscores.items()})
            return float(score)
        return float(self._score(pose, computed=computed))

    def batch_scores(self, poses: Poses | NDArray) -> NDArray[np.float64]:
        """Score many poses at once.

        Mirrors ``get_score``: the dependencies are computed once for the whole batch, then
        ``_batch_scores`` scores the poses. By default that calls ``_score`` for every pose; a
        scoring function that can do better overrides ``_batch_scores``, not this method.

        How much faster this is than scoring the poses one at a time depends on the scoring
        function. Cheap terms and grids, where the cost of a call is a large part of the work, are
        several times faster in a batch. Terms between ligand and receptor are about as fast
        either way: their work is the neighbor search and the kernel for every atom pair, which a
        batch does not reduce.

        Parameters
        ----------
        poses : Poses, ndarray
            The poses to score, or their raw values, shape ``(n_poses, n_dims)``.

        Returns
        -------
        NDArray
            One score per pose, same order as `poses`.

        See Also
        --------
        get_score : Score one pose.
        pyrite.Poses : A batch of poses.

        Examples
        --------
        >>> scores = scoring_function.batch_scores(poses)
        >>> best = poses[np.argmin(scores)]
        """
        realized = Realization(poses, batched=True)
        computed_batch = _NarrowingComputed(
            {dep: dep.compute(realized) for dep in self._resolved_dependencies()}
        )
        return self._batch_scores(poses, computed_batch)

    def get_score_and_gradient(self, pose: Pose | NDArray) -> tuple[float, NDArray[np.float64]]:
        """The score of a pose and its gradient with respect to the pose variables.

        For a local optimizer: ``minimize(lambda v: sf.get_score_and_gradient(v), x0, jac=True)``.
        Terms that know their derivative (``GridScore``, ``InternalOverlap``) compute it
        analytically; every other term falls back to central finite differences of its own score,
        so this works for any scoring function, and composites combine the parts by the chain rule.

        Parameters
        ----------
        pose : Pose, ndarray
            The pose, or its raw values.

        Returns
        -------
        score : float
            Equal to ``get_score(pose)``.
        gradient : ndarray
            Shape ``(n_dims,)``: the derivative of the score with respect to every pose variable,
            in the layout of the pose.

        See Also
        --------
        get_score : Only the score.
        pyrite.search.BasinHopping : Takes this function with ``jac=True``.
        pyrite.Mol.pose_gradient : For terms with an analytic gradient.

        Examples
        --------
        >>> score, gradient = scoring_function.get_score_and_gradient(pose)
        >>> from scipy.optimize import minimize
        >>> result = minimize(
        ...     lambda v: scoring_function.get_score_and_gradient(Pose(v, ligand.layout)),
        ...     np.asarray(pose),
        ...     jac=True,
        ...     method="L-BFGS-B",
        ... )
        """
        realized = Realization(pose, batched=False)
        computed = _NarrowingComputed(
            {dep: dep.compute(realized) for dep in self._resolved_dependencies()}
        )
        score, gradient = self._score_and_gradient(pose, computed)
        return float(score), np.asarray(gradient, dtype=np.float64)

    _GRADIENT_STEP = 1e-6  # of the central differences, in Angstrom and radians

    def _score_and_gradient(self, pose, computed) -> tuple[float, NDArray[np.float64]]:
        """The score and its gradient with respect to the pose. Optional.

        .. note::
            Do not call this method directly. Use ``get_score_and_gradient`` instead.

        :meta public:

        The default is central finite differences of this term's own score (two scores per pose
        variable): correct for every term, not fast. Implement it for a term whose gradient is
        known; with the gradient per atom position (``dS/dx``), ``self.mol.pose_gradient`` gives
        the gradient with respect to the pose.

        Parameters
        ----------
        pose : Pose, ndarray
            The pose for which to calculate the score and gradient.
        computed : dict[Dependency, Any]
            The computed dependencies, as for ``_score``.

        Returns
        -------
        score : float
            The score of `pose`.
        gradient : ndarray
            The gradient with respect to the pose variables, of shape ``(n_dims,)``.

        See Also
        --------
        get_score_and_gradient : The score of a pose and its gradient.
        pyrite.Mol.pose_gradient : Turn a gradient per atom into a gradient per pose variable.

        Examples
        --------
        >>> def _score_and_gradient(self, pose, computed):
        ...     positions = self.mol.pose_to_positions(pose)
        ...     offsets = positions - self.target
        ...     score = float((offsets**2).sum())
        ...     forces = 2 * offsets  # dS/dx of every atom
        ...     return score, self.mol.pose_gradient(pose, positions, forces)
        """
        values = np.asarray(pose, dtype=np.float64)
        layout = getattr(pose, "layout", None)
        gradient = np.empty(len(values))
        for k in range(len(values)):
            step = np.zeros(len(values))
            step[k] = self._GRADIENT_STEP
            up, down = values + step, values - step
            if layout is not None:
                up, down = Pose(up, layout), Pose(down, layout)
            gradient[k] = (self.get_score(up) - self.get_score(down)) / (2 * self._GRADIENT_STEP)
        return self._score(pose, computed), gradient

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
            The computed dependencies of the whole batch, supplied by ``batch_scores``: like
            `computed` of ``_score``, with a leading ``n_poses`` axis on everything. The default
            calls ``_score`` for every pose, on ``computed_batch.row(i)``, so the dependencies are
            still computed once for the whole batch.

        Returns
        -------
        numpy.ndarray
            One score per pose, of shape ``(n_poses,)``.

        See Also
        --------
        batch_scores : Score many poses at once.
        _score : Score one pose.

        Examples
        --------
        >>> def _batch_scores(self, poses, computed_batch):
        ...     positions = self.mol.pose_to_positions(poses)  # (n_poses, n_atoms, 3)
        ...     return np.linalg.norm(positions - self.target, axis=2).sum(axis=1)
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
            A scoring function that clamps the scores of this one.

        See Also
        --------
        pyrite.scoring.Clamp : The same, as a class.

        Examples
        --------
        >>> repulsion = Repulsion(ligand, receptor).clamp(max_score=10.0)
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
            The score of `pose`.

        See Also
        --------
        get_score : Score a pose.
        _batch_scores : Score many poses at once (optional).
        _score_and_gradient : The score and its gradient (optional).

        Examples
        --------
        >>> def _score(self, pose, computed):
        ...     positions = self.mol.pose_to_positions(pose)
        ...     return float(np.linalg.norm(positions - self.target, axis=1).sum())
        """

    def get_dependencies(self) -> list[Dependency]:
        """Get the dependencies of this scoring function.

        A composite returns the dependencies of all its terms, in a ``list``. Two dependencies
        that compare equal are merged into one computation, but can still need different results
        back (another ``k`` or cutoff, see
        :meth:`~pyrite.scoring.dependencies.Dependency.narrow`): a ``set`` would silently keep
        only one of them.

        Returns
        -------
        list of Dependency
            The dependencies, possibly with ones that compare equal.

        See Also
        --------
        pyrite.scoring.dependencies.Dependency : Shared, expensive computations.

        Examples
        --------
        >>> def get_dependencies(self):
        ...     return [self.nn_dep]
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

    def _score_and_gradient(self, pose, computed):
        total, gradient = 0.0, 0.0
        for func in self.funcs:
            # pylint: disable=protected-access
            score, partial = func._score_and_gradient(pose, computed)
            total += score
            gradient = gradient + partial
        return total, gradient

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
        """Score points of a given atom type, see ``_KNNScoringFunction._score_field``.

        :meta public:

        Parameters
        ----------
        r, idx : numpy.ndarray
            The distances to, and indices of, the nearest neighbors of every point.
        atom_type : AtomType or numpy.ndarray
            One atom type for all points, or one per point.

        Returns
        -------
        numpy.ndarray
            The score of every point.

        See Also
        --------
        pyrite.scoring.grid.GridScore : Evaluates this on the vertices of a grid.

        Examples
        --------
        >>> scores = composite._score_field(r, idx, AtomType.NitrogenAcceptor)
        """
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

    def _score_and_gradient(self, pose, computed):
        # pylint: disable=protected-access
        n_dims = len(np.asarray(pose))

        def part(side):
            if isinstance(side, ScoringFunction):
                return side._score_and_gradient(pose, computed)
            return side, np.zeros(n_dims)

        (a, da), (b, db) = part(self.left), part(self.right)
        match self.operator:
            case "*":
                return a * b, da * b + a * db
            case "/":
                return a / b, (da * b - a * db) / b**2
            case "^":
                value = a**b
                gradient = np.zeros(n_dims)
                if np.any(da):  # d(a^b)/da = b a^(b-1)
                    gradient = gradient + b * a ** (b - 1) * da
                if np.any(db):  # d(a^b)/db = a^b ln a
                    gradient = gradient + value * np.log(a) * db
                return value, gradient
        raise ValueError(f"Unknown operator {self.operator!r}")

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
        """Score points of a given atom type, see ``_KNNScoringFunction._score_field``.

        :meta public:

        Parameters
        ----------
        r, idx : numpy.ndarray
            The distances to, and indices of, the nearest neighbors of every point.
        atom_type : AtomType or numpy.ndarray
            One atom type for all points, or one per point.

        Returns
        -------
        numpy.ndarray
            The score of every point.

        See Also
        --------
        pyrite.scoring.grid.GridScore : Evaluates this on the vertices of a grid.

        Examples
        --------
        >>> scores = composite._score_field(r, idx, AtomType.NitrogenAcceptor)
        """
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

    Use it to keep one term from dominating a composite, for example a repulsion that explodes when
    two atoms overlap. The gradient is zero where the score is clamped.

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

    See Also
    --------
    ScoringFunction.clamp : The same, as a method.

    Examples
    --------
    >>> from pyrite.scoring import Clamp, Repulsion
    >>> repulsion = Clamp(Repulsion(ligand, receptor), max_score=10.0)
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

    def _score_and_gradient(self, pose, computed):
        # pylint: disable=protected-access
        score, gradient = self.scoring_function._score_and_gradient(pose, computed)
        if score < self.min_score or score > self.max_score:
            return float(np.clip(score, self.min_score, self.max_score)), np.zeros_like(gradient)
        return score, gradient

    def _score_and_store(self, pose, computed, subscores) -> float:
        score = np.clip(
            self.scoring_function._score_and_store(pose, computed=computed, subscores=subscores),
            self.min_score,
            self.max_score,
        )
        subscores[self] = score
        return score

    def _score_field(self, r, idx, atom_type):
        """Score points of a given atom type, see ``_KNNScoringFunction._score_field``.

        :meta public:

        Parameters
        ----------
        r, idx : numpy.ndarray
            The distances to, and indices of, the nearest neighbors of every point.
        atom_type : AtomType or numpy.ndarray
            One atom type for all points, or one per point.

        Returns
        -------
        numpy.ndarray
            The score of every point.

        See Also
        --------
        pyrite.scoring.grid.GridScore : Evaluates this on the vertices of a grid.

        Examples
        --------
        >>> scores = composite._score_field(r, idx, AtomType.NitrogenAcceptor)
        """
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

    See Also
    --------
    ScoringFunction : Scoring functions are combined with ``+``, ``-``, ``*``, ``/`` and ``**``.

    Examples
    --------
    >>> from pyrite.scoring import ConstantTerm, Gaussian
    >>> shifted = Gaussian(ligand, receptor) + 1.5  # makes a ConstantTerm(1.5)
    >>> explicit = Gaussian(ligand, receptor) + ConstantTerm(1.5)
    """

    def __init__(self, constant: float):
        self.constant = constant

    def _score(self, *args, **kwargs) -> float:
        return self.constant

    def _score_and_gradient(self, pose, computed):
        return self.constant, np.zeros(len(np.asarray(pose)))

    def _score_field(self, r, idx, atom_type):
        """Score points of a given atom type, see ``_KNNScoringFunction._score_field``.

        :meta public:

        Parameters
        ----------
        r, idx : numpy.ndarray
            The distances to, and indices of, the nearest neighbors of every point.
        atom_type : AtomType or numpy.ndarray
            One atom type for all points, or one per point.

        Returns
        -------
        numpy.ndarray
            The score of every point.

        See Also
        --------
        pyrite.scoring.grid.GridScore : Evaluates this on the vertices of a grid.

        Examples
        --------
        >>> scores = composite._score_field(r, idx, AtomType.NitrogenAcceptor)
        """
        return np.full(len(r), self.constant)

    def _batch_scores(self, poses, computed_batch) -> NDArray[np.float64]:
        return np.full(len(poses), self.constant)


class _RDKitScoringFunction(ScoringFunction, ABC):
    """Base class for scoring functions that need a real RDKit conformer of the molecule.

    Implement ``_score(pose, computed)`` as for any scoring function, and read the posed molecule
    with ``computed[self.rdkit_dep]``: an :class:`rdkit.Chem.rdchem.Mol` copy of ``self.mol`` with
    the pose as its only conformer (``confId=-1``), which is private to this call and may be used
    freely. It is made by a shared :class:`~pyrite.scoring.dependencies.RDKitDependency`, so any
    number of these terms in one composite share one copy per pose, and ``self.mol`` is never
    modified.

    Parameters
    ----------
    mol : Mol
        The molecule to make a conformer of.

    See Also
    --------
    pyrite.scoring.dependencies.RDKitDependency : Makes the posed copy.
    pyrite.Mol.to_rdkit : An RDKit copy with a pose, outside scoring.

    Examples
    --------
    >>> from rdkit.Chem import Descriptors3D
    >>> class RadiusOfGyrationRDKit(_RDKitScoringFunction):
    ...     def _score(self, pose, computed):
    ...         posed = computed[self.rdkit_dep]
    ...         return Descriptors3D.RadiusOfGyration(posed)
    """

    def __init__(self, mol: Mol):
        self.mol = mol
        self.rdkit_dep = RDKitDependency(mol)

    def get_dependencies(self) -> list[Dependency]:
        return [self.rdkit_dep]
