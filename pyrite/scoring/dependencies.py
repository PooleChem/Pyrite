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
from collections.abc import Callable
from typing import Any

import numpy as np
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

    def narrow(self, computed: Any) -> Any:
        """Narrow a computed result down to what *this* dependency instance asked for.

        Dependencies that share a ``group_key`` are merged (see :meth:`merge_all`)
        into one shared computation, run at whichever parameters cover every member
        of the group (e.g. the widest ``k``/cutoff among them, for
        :class:`KNNDependency`) — so the raw result handed back by that shared
        computation can be wider than what any *individual* dependency in the group
        actually needs. ``narrow`` is called with that raw, possibly-wider result
        and should return the view specific to ``self`` — the default here is the
        identity (nothing to narrow), correct for any dependency type that doesn't
        have this "computed once, wide; used many times, narrower" shape.

        .. note::
            This must be resolved per dependency *instance*, not once per merged
            group — two dependencies in the same group can have the same
            ``group_key`` (so they merge and share one computation) while still
            wanting different narrowed views back (e.g. different ``k``/cutoff).
            A plain ``dict`` can't hold two different values under two keys that
            compare equal, so this can't be precomputed into a dict keyed by
            ``group_key`` — it has to be called per instance, on demand (see
            ``_NarrowingComputed``).

        Parameters
        ----------
        computed : Any
            The raw result from ``compute``/``compute_batch``, for this
            dependency's merged group.

        Returns
        -------
        Any
        """
        return computed

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
        return isinstance(other, type(self)) and other.group_key(other) == self.group_key(self)

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


class _NarrowingComputed:
    """Wraps a ``{Dependency: raw_result}`` dict so ``computed[dep]`` returns
    ``dep``'s own narrowed view (:meth:`Dependency.narrow`) rather than the
    raw, possibly wider-scoped result a merged group actually computed.

    This has to be resolved lazily, per lookup, not precomputed into a plain
    dict — two dependencies in the same merged group compare equal (that's
    what lets them share one computation) but can still want different
    narrowed views back, and a plain dict can't hold two different values
    under two keys that compare equal. Narrowing on each ``__getitem__`` call,
    using the exact instance passed in (not its equivalence class), is what
    makes ``computed[self.nn_dep]`` correct for every dependent instance
    sharing a merged group, not just whichever happened to be computed last.

    Used by :meth:`~pyrite.scoring.ScoringFunction.get_score`/``batch_scores``
    — not something a scoring function author constructs directly.
    """

    def __init__(self, raw: dict):
        self._raw = raw

    def __getitem__(self, dep: Dependency):
        return dep.narrow(self._raw[dep])


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
        point_cloud: NDArray,
        query_f: Callable[[int], NDArray],
        k: int,
        distance_upper_bound: float,
    ):  # pylint: disable=too-many-arguments
        self.point_cloud = point_cloud
        self.querying = query_f
        self.k = k
        self.distance_upper_bound = distance_upper_bound

        # Get tree key unique to point_cloud
        pc_meta = (self.point_cloud.shape, str(self.point_cloud.dtype))
        pc_data = self.point_cloud.tobytes()
        self.tree_hash = hash((pc_meta, pc_data))

        # Initialize tree
        self.tree = KDTreeCache.get_tree(self.tree_hash, self.point_cloud)

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
        r : NDArray
            The distances to the nearest neighbors, shape ``(n_points, k)``.
        idx : NDArray
            The indices of the nearest neighbors, shape ``(n_points, k)``.
        mask : NDArray
            A boolean mask indicating which neighbors are valid, shape ``(n_points, k)``.
        """
        r, idx = self.tree.query(
            self.querying(conf_id),
            k=self.k,
            distance_upper_bound=self.distance_upper_bound,
        )
        if self.k == 1:
            # SciPy squeezes the neighbour axis for k == 1. Keep it, as compute_batch does, so
            # that the last axis is always the neighbours (and `narrow` slices the right axis).
            r, idx = r[..., None], idx[..., None]
        return r, idx, (idx != self.tree.n)

    def compute_batch(
        self, conf_ids
    ) -> tuple[
        NDArray,
        NDArray,
        NDArray,
    ]:
        """Execute the nearest neighbor search for many conformers at once.

        Batched equivalent of ``compute`` — queries the tree once for all
        `conf_ids` instead of once per conformer.

        Parameters
        ----------
        conf_ids : array_like[int]
            The conformer ids from which to retrieve the points to query on.

        Returns
        -------
        r : NDArray
            Shape ``(n_conf_ids, n_points, k)``.
        idx : NDArray
            Shape ``(n_conf_ids, n_points, k)``.
        mask : NDArray
            Shape ``(n_conf_ids, n_points, k)``.
        """
        positions = np.stack([self.querying(conf_id) for conf_id in conf_ids])
        n_conf_ids, n_points, _ = positions.shape

        r, idx = self.tree.query(
            positions.reshape(-1, 3),
            k=self.k,
            distance_upper_bound=self.distance_upper_bound,
        )
        r = r.reshape(n_conf_ids, n_points, self.k)
        idx = idx.reshape(n_conf_ids, n_points, self.k)
        return r, idx, (idx != self.tree.n)

    def narrow(self, computed):
        """Narrow a merged group's shared query result down to this instance's
        own `k`/`distance_upper_bound`.

        The merged/shared query (see `merge_group`) runs at the *widest* `k` and
        `distance_upper_bound` across every dependency in the group, so a member
        with a smaller `k` gets back extra, farther-out neighbor columns it never
        asked for, and a member with a smaller cutoff gets back neighbors beyond
        its own configured distance. Both need trimming back down per instance —
        slicing to `self.k` neighbor columns (already sorted nearest-first by the
        KDTree query, so this keeps exactly the `k` nearest) and masking out
        anything beyond `self.distance_upper_bound`.

        Parameters
        ----------
        computed : tuple[NDArray, NDArray, NDArray]
            The group's raw `(r, idx, mask)`, shape `(..., k_group)` on the last
            axis, `k_group` >= `self.k`.

        Returns
        -------
        tuple[NDArray, NDArray, NDArray]
        """
        r, idx, mask = computed
        r, idx, mask = r[..., : self.k], idx[..., : self.k], mask[..., : self.k]
        mask = mask & (r < self.distance_upper_bound)
        return r, idx, mask

    @classmethod
    def group_key(cls, dep):
        """Returns the group_key identifier of a dependency.

        The ``group_key`` of ``KNNDependency`` is a tuple containing the `tree_hash` and `querying`.

        Parameters
        ----------
        dep : Dependency
            The dependency to get the group key identifier for.


        Returns
        -------
        tuple
        """
        return dep.tree_hash, dep.querying

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
            best_k.point_cloud,
            best_k.querying,
            best_k.k,
            best_ub.distance_upper_bound,
        )

    # Deliberately excludes k/distance_upper_bound — that's what lets two
    # dependencies with different k/cutoff still merge into one shared query
    # (at the widest of the two). Each instance still gets its own correct view
    # back via narrow(), so this doesn't lose per-instance correctness — see
    # narrow()'s docstring for why that has to happen per instance rather than
    # being resolved here.
    def __hash__(self):
        return hash((KNNDependency, self.tree_hash, self.querying))

    def __eq__(self, other):
        return isinstance(other, KNNDependency) and other.group_key(other) == self.group_key(self)
