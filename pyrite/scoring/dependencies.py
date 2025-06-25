"""

======================================================
Dependency module (:mod:`pyrite.scoring.dependencies`)
======================================================

.. currentmodule:: pyrite.scoring.dependencies

Tools for defining dependencies for :class:`~pyrite.scoring._base.ScoringFunction`, such that
expensive operations are only executed once.

This module provides utilities for defining :class:`Dependency`. A :class:`Dependency` can execute
operations via its :meth:`~Dependency.compute` method.
A :class:`~pyrite.scoring._base.ScoringFunction` can register dependencies by adding them to its
:meth:`~pyrite.scoring.ScoringFunction.get_dependencies`. This dependency is then hashed,
based on :meth:`~Dependency.group_key`, in such a way that only dependencies that execute a similar
enough computation are combined, and thus only executed once.

Classes
-------

.. autosummary::
   :toctree: generated/

   Dependency
   KNNDependency
   KDTreeCache


See Also
--------
~pyrite.scoring._base.ScoringFunction : The ``ScoringFunction`` class.


Notes
-----
To subclass ``Dependency``, the following methods should be implemented:

:meth:`~Dependency.compute(conf_id)`
    In this method the expensive operation should be executed.
:meth:`~Dependency.group_key(dep)`
    This method should map a group of dependencies to the same key, i.e., all dependencies that
    return the same `group_key` are combined by :meth:`~Dependency.merge_group`.
:meth:`~Dependency.merge_group(deps)`
    This method is responsible for merging a group of dependencies. The input is a group of
    dependencies that are deemed equal by :meth:`~Dependency.group_key`. The output should be
    a single dependency.

"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable

from numpy.typing import NDArray
from scipy.spatial import KDTree


class Dependency(ABC):
    """
    Abstract Dependency class.

    The :class:`Dependency` class can be used to execute expensive operations only once,
    and reuse the result in multiple scoring functions.

    A :class:`~pyrite.scoring.ScoringFunction` can register a dependency by calling
    :meth:`~pyrite.scoring.ScoringFunction.get_dependencies`.
    This dependency is then hashed, based on :meth:`group_key`, in such a way that only dependencies
    that execute a similar enough computation are combined, and thus only executed once.

    .. note::
        This is an abstract class, and specific implementation can thus vary between
        implementations. To implement a new :class:`Dependency`,
        please refer to :mod:`~pyrite.scoring.dependencies`.


    """

    @abstractmethod
    def compute(self, conf_id: int) -> Any:
        """Compute ``self`` based on the supplied `conf_id`.

        Parameters
        ----------
        conf_id : int
            The conformer id to use in computation.

        Returns
        -------
        Any
        """
        pass

    @classmethod
    @abstractmethod
    def group_key(cls, dep):
        """Return the group key identifier of a specific dependency.

        This identifier should be unique to a group of dependencies, where the group
        is defined by all the dependencies that perform the same computation and
        should thus be merged.

        Parameters
        ----------
        dep : Dependency
            The dependency for which to get the group key.

        Returns
        -------
        object
        """
        pass

    @classmethod
    @abstractmethod
    def merge_group(cls, deps):
        """Merge a group of dependencies.

        This method should return a single ``Dependency`` instance based on a group of dependencies.
        The returned instance should be the instance that computes all the data needed for all
        of the dependencies in the group.

        Parameters
        ----------
        deps : array_like[Dependency]
            The list of dependencies to merge.

        Returns
        -------
        Dependency
        """
        pass

    def __hash__(self):
        return hash((type(self), self.group_key(self)))

    def __eq__(self, other):
        return isinstance(other, type(self)) and other.group_key(
            other
        ) == self.group_key(self)

    @classmethod
    def merge_all(cls, deps):
        """Merge a list of dependencies based on ``group_key``.

        This method will group the dependencies based on ``group_key`` and call ``merge_group`` on
        each group, to make sure that for each group, only one ``Dependency`` instance is computed.

        Parameters
        ----------
        deps : array_like[Dependency]
            The list of dependencies to merge.

        Returns
        -------
        Set[Dependency]
        """
        type_key = defaultdict(list)
        for dep in deps:
            type_key[(type(dep), dep.group_key(dep))].append(dep)

        merged = set()
        for (dep_cls, _), group in type_key.items():
            merged.add(dep_cls.merge_group(group))
        return merged


class KDTreeCache:  # pylint: disable=too-few-public-methods
    """
    Cache object holding KDTrees.

    Can be used to store KDTrees and access them globally.

    """

    _trees: dict[int | str, KDTree] = {}

    @classmethod
    def get_tree(cls, key: int | str, point_cloud: NDArray = None) -> KDTree:
        """Retrieve or create the tree selected by the `key`.

        When a tree with `key` does not yet exist, a new one is created based on `point_cloud`.

        Parameters
        ----------
        key : int, str
            The key by which to select the tree.

        point_cloud : NDArray, optional
            Point cloud used to create a new tree if `key` does not exist.

        Returns
        -------
        KDTree
        """
        tree = cls._trees.get(key)
        if tree is None:
            tree = KDTree(point_cloud)
            cls._trees[key] = tree
        return tree


class KNNDependency(Dependency):
    """
    k-Nearest Neighbors Dependency.

    This :class:`Dependency` is used to retrieve the nearest neighbors of a given list of points,
    based on a :class:`~scipy.spatial.KDTree` build from a point cloud.

    The :class:`Dependency` class can be used to execute expensive operations only once and
    reuse the result in multiple scoring functions.

    A :class:`~pyrite.scoring.ScoringFunction` can register a dependency by calling
    :meth:`~pyrite.scoring.ScoringFunction.get_dependencies`.
    This dependency is then hashed, based on :meth:`group_key`, in such a way that only dependencies
    that execute a similar enough computation are combined, and thus only executed once.


    Parameters
    ----------
    tree_id : int, str
        A unique identifier for the point cloud used in the dependency. This is
        combined with the `query_f` to create the ``group_key``.
    point_cloud : NDArray
        The point cloud used to build the ``KDTree``.
    query_f : Callable[[int], NDArray]
        A callable function that is used to get the points to query on the ``KDTree``. This is
        combined with the `tree_id` to create the `group_key`.
    k : int
        The number of neighbors to retrieve.
    distance_upper_bound : float
        The upper bound of distance to consider when retrieving neighbors.


    """

    def __init__(
        self,
        tree_id: int | str,
        point_cloud: NDArray,
        query_f: Callable[[int], NDArray],
        k: int,
        distance_upper_bound: float,
        # q_mask: NDArray = None,
    ):  # pylint: disable=too-many-arguments
        self.point_cloud = point_cloud
        self.querying = query_f
        self.k = k
        self.distance_upper_bound = distance_upper_bound
        self.tree_id = tree_id
        # if q_mask is None:
        #     q_mask = np.ones(len(self.querying(-1)), dtype=bool)
        # self.q_mask = q_mask
        # # self.q_mask = (
        # #     q_mask
        # #     if q_mask is not None
        # #     else np.ones(len(self.querying(-1)), dtype=bool)
        # # )

        # Initialize tree
        self.tree = KDTreeCache.get_tree(self.tree_id, self.point_cloud)

    def compute(
        self, conf_id
    ) -> tuple[
        float | NDArray,
        int | NDArray,
        bool | NDArray,
    ]:
        """Execute the nearest neighbor search.

        Parameters
        ----------
        conf_id :
            The conformer_id from which to retrieve the points to query on.

        Returns
        -------
        r : float, NDArray
            The distances to the nearest neighbors.
        idx : int, NDArray
            The indices of the nearest neighbors.
        mask : NDArray
            A boolean mask indicating which neighbors are valid.
        """
        # TODO: distance upper bound!
        # print("dep", self.tree.n)
        r, idx = self.tree.query(
            self.querying(conf_id), k=self.k, distance_upper_bound=8
        )
        return r, idx, (idx != self.tree.n)

    @classmethod
    def group_key(cls, dep):
        """Returns the group_key identifier of a dependency.

        The ``group_key`` of ``KNNDependency`` is a tuple containing the `tree_id` and `querying`.

        Parameters
        ----------
        dep : Dependency
            The dependency to get the group key identifier for.


        Returns
        -------
        tuple
        """
        return dep.tree_id, dep.querying

    @classmethod
    def merge_group(cls, deps):
        """Merge a group of dependencies.

        This method returns a new ``KNNDependency`` instance with the merged properties of the
        group.
        It selects the highest `k` and `distance_upper_bound` among the dependencies in the group.

        Parameters
        ----------
        deps : array_like[Dependency]
            The group of dependencies to merge.

        Returns
        -------
        Dependency
        """
        best_k = max(deps, key=lambda d: d.k)
        best_ub = max(deps, key=lambda d: d.distance_upper_bound)
        return cls(
            best_k.tree_id,
            best_k.point_cloud,
            best_k.querying,
            best_k.k,
            best_ub.distance_upper_bound,
        )

    # TODO: cutoff?
    def __hash__(self):
        return hash(
            (
                KNNDependency,
                self.tree_id,
                self.querying,
                # self.k,
                # self.distance_upper_bound,
            )
        )

    # TODO: cutoff?
    def __eq__(self, other):
        return (
            isinstance(other, KNNDependency)
            and other.group_key(other) == self.group_key(self)
            # and self.k == other.k
            # and self.distance_upper_bound == other.distance_upper_bound
        )
