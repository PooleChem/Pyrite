"""

=================================================
Dependencies (:mod:`pyrite.scoring.dependencies`)
=================================================

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
   PositionDependency
   RDKitDependency
   Realization
   PositionQuery
   KDTreeCache


See Also
--------
~pyrite.scoring._base.ScoringFunction : The ``ScoringFunction`` class.


Notes
-----
To subclass ``Dependency``, the following methods should be implemented:

:meth:`~Dependency.compute`
    In this method the expensive operation should be executed. `realized` is a
    :class:`Realization` of the pose (or batch of poses) being scored, which hands out the atom
    positions (:meth:`Realization.positions`) or a private RDKit copy
    (:meth:`Realization.rdkit`) of any :class:`~pyrite.Mol`, computed at most once per
    scoring call.
:meth:`~Dependency.group_key`
    This method should map a group of dependencies to the same key, i.e., all dependencies that
    return the same `group_key` are combined by :meth:`~Dependency.merge_group`.
:meth:`~Dependency.merge_group`
    This method is responsible for merging a group of dependencies. The input is a group of
    dependencies that are deemed equal by :meth:`~Dependency.group_key`. The output should be
    a single dependency.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from scipy.spatial import KDTree

from .._common import Mol


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

    See Also
    --------
    KNNDependency : A shared nearest neighbor search.
    PositionDependency : The atom positions of a molecule.
    RDKitDependency : An RDKit copy of a molecule with the pose as its conformer.

    Examples
    --------
    >>> class CentroidDependency(Dependency):
    ...     def __init__(self, mol):
    ...         self.mol = mol
    ...
    ...     def compute(self, realized):
    ...         return realized.positions(self.mol).mean(axis=-2)
    ...
    ...     @classmethod
    ...     def group_key(cls, dep):
    ...         return dep.mol  # one computation per molecule
    ...
    ...     @classmethod
    ...     def merge_group(cls, deps):
    ...         return deps[0]
    """

    @abstractmethod
    def compute(self, realized: Realization) -> Any:
        """Compute ``self`` for the pose, or batch of poses, in `realized`.

        Written to work for both: a batch simply has a leading ``n_poses`` axis on everything.

        Parameters
        ----------
        realized : Realization
            The pose (or poses) being scored.

        Returns
        -------
        Any
            The result, which scoring functions read as ``computed[dependency]``.

        See Also
        --------
        Realization : The pose(s) being scored.
        narrow : The part of a merged result that one dependency asked for.

        Examples
        --------
        >>> def compute(self, realized):
        ...     return realized.positions(self.mol).mean(axis=-2)
        """
        pass

    def narrow(self, computed: Any) -> Any:
        """Narrow a computed result down to what this dependency asked for.

        Dependencies that compare equal are merged into one computation, at parameters that
        cover all of them (for :class:`KNNDependency`: the largest ``k`` and cutoff). Every
        dependency then gets its own view of that result back through this method. The default
        returns the result unchanged, which is right for dependencies without such parameters.

        Parameters
        ----------
        computed : Any
            The result of ``compute`` for the merged group.

        Returns
        -------
        Any
            The part of `computed` that this dependency asked for.

        See Also
        --------
        KNNDependency.narrow : Keeps a term's own `k` and cutoff.

        Examples
        --------
        >>> def narrow(self, computed):
        ...     return computed[..., : self.k]
        """
        return computed

    def row(self, computed: Any, i: int) -> Any:
        """The result for pose `i` of a batch, as ``compute`` would have given it for that pose.

        ``compute`` puts a leading ``n_poses`` axis on everything for a batch, so the default
        takes entry `i` of it (of every part of a tuple). Override it only for a dependency whose
        batched result is not laid out like that.

        Parameters
        ----------
        computed : Any
            The result of ``compute`` for a batch.
        i : int
            The index of the pose.

        Returns
        -------
        Any
            The result for pose `i`.

        See Also
        --------
        pyrite.scoring.ScoringFunction._batch_scores : Its default uses this.

        Examples
        --------
        >>> first = dependency.row(batch_result, 0)
        """
        if isinstance(computed, tuple):
            return tuple(part[i] for part in computed)
        return computed[i]

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

        See Also
        --------
        merge_all : Groups dependencies by this key.

        Examples
        --------
        >>> @classmethod
        ... def group_key(cls, dep):
        ...     return dep.mol
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

        See Also
        --------
        merge_all : Calls this for every group.
        narrow : Gives every dependency its own part back.

        Examples
        --------
        >>> @classmethod
        ... def merge_group(cls, deps):
        ...     widest = max(deps, key=lambda dep: dep.k)
        ...     return cls(widest.mol, k=widest.k)
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

        See Also
        --------
        group_key : Decides which dependencies are merged.
        merge_group : Merges one group.

        Examples
        --------
        >>> deps = scoring_function.get_dependencies()
        >>> merged = Dependency.merge_all(deps)  # one per computation
        """
        type_key = defaultdict(list)
        for dep in deps:
            type_key[(type(dep), dep.group_key(dep))].append(dep)

        merged = set()
        for (dep_cls, _), group in type_key.items():
            merged.add(dep_cls.merge_group(group))
        return merged


class _NarrowingComputed:
    """Hand every dependency its own view of the merged results, see :meth:`Dependency.narrow`.

    ``computed[dep]`` narrows on every lookup, for the exact instance asked for: a plain ``dict``
    cannot, as dependencies that compare equal are the same key. Made by
    :meth:`~pyrite.scoring.ScoringFunction.get_score` and ``batch_scores``; a scoring function
    only reads from it.

    Parameters
    ----------
    raw : dict[Dependency, Any]
        The result of every merged group, keyed by its representative.

    See Also
    --------
    Dependency.narrow : Called on every lookup.

    Examples
    --------
    >>> r, idx, mask = computed[self.nn_dep]  # in a scoring function's _score
    """

    def __init__(self, raw: dict):
        self._raw = raw

    def __getitem__(self, dep: Dependency):
        return dep.narrow(self._raw[dep])

    def row(self, i: int) -> _NarrowingComputed:
        """Return the computed dependencies of pose `i` of a batch, as ``get_score`` sees them.

        Used by the default ``_batch_scores``, which scores a batch one pose at a time on the
        results computed once for the whole batch.

        Parameters
        ----------
        i : int
            The index of the pose in the batch.

        Returns
        -------
        _NarrowingComputed
            The computed dependencies of that pose.

        See Also
        --------
        Dependency.row : The result of one dependency for one pose.

        Examples
        --------
        >>> scores = [self._score(pose, computed_batch.row(i)) for i, pose in enumerate(poses)]
        """
        return _NarrowingComputed({dep: dep.row(value, i) for dep, value in self._raw.items()})


class Realization:
    """One pose, or a batch of poses, being scored — realised lazily per molecule.

    Created by :meth:`~pyrite.scoring.ScoringFunction.get_score` and ``batch_scores`` and handed
    to :meth:`Dependency.compute`. It is the only place where a pose turns into something
    concrete, and does so at most once per :class:`~pyrite.Mol`, however many dependencies ask:

    - :meth:`positions`: the atom positions, computed in numpy by
      :meth:`~pyrite.Mol.pose_to_positions`.
    - :meth:`rdkit`: a private RDKit copy of the molecule with the pose as its conformer, for
      terms that need RDKit itself.

    Nothing is ever written to the :class:`~pyrite.Mol` that is being scored, so any number of
    threads can score poses of one ``Mol`` at the same time, and there is nothing to clean up.

    Parameters
    ----------
    poses : Pose, Poses, ndarray
        The pose(s) to realise.
    batched : bool
        Whether `poses` is a batch (``Poses``, or a 2D array), rather than one pose.

    See Also
    --------
    Dependency.compute : Receives this.
    pyrite.scoring.ScoringFunction.get_score : Creates this.

    Examples
    --------
    >>> realized = Realization(pose, batched=False)
    >>> realized.positions(ligand).shape  # (n_atoms, 3)
    """

    def __init__(self, poses, batched: bool):
        self.poses = poses
        self.batched = batched
        self._positions: dict[Mol, NDArray] = {}
        self._rdkit: dict[Mol, Chem.Mol | list[Chem.Mol]] = {}

    def positions(self, mol: Mol) -> NDArray:
        """Return the atom positions of `mol` for the pose(s), computed once per molecule.

        Every dependency that asks for the same molecule gets the same array, so it is computed once
        per call.

        Parameters
        ----------
        mol : Mol
            The molecule the pose(s) belong to.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_atoms, 3)``, or ``(n_poses, n_atoms, 3)`` for a batch.

        See Also
        --------
        rdkit : An RDKit copy with the pose.
        pyrite.Mol.pose_to_positions : Computes the positions.

        Examples
        --------
        >>> positions = realized.positions(ligand)
        """
        if mol not in self._positions:
            self._positions[mol] = mol.pose_to_positions(self.poses)
        return self._positions[mol]

    def rdkit(self, mol: Mol) -> Chem.Mol | list[Chem.Mol]:
        """Return a private RDKit copy of `mol` with the pose as its only conformer.

        The copies are the caller's to use freely: they are not shared with anything else, and the
        molecule being scored is not touched.

        Parameters
        ----------
        mol : Mol
            The molecule the pose(s) belong to.

        Returns
        -------
        rdkit.Chem.rdchem.Mol or list of rdkit.Chem.rdchem.Mol
            One copy for a single pose, or a list of them for a batch.

        See Also
        --------
        positions : Only the positions, without RDKit.

        Examples
        --------
        >>> posed = realized.rdkit(ligand)
        >>> posed.GetConformer().GetPositions().shape  # (n_atoms, 3)
        """
        if mol not in self._rdkit:
            positions = self.positions(mol)
            self._rdkit[mol] = (
                [mol._rdkit_with_positions(p) for p in positions]
                if self.batched
                else mol._rdkit_with_positions(positions)
            )
        return self._rdkit[mol]


class PositionQuery:
    """A callable ``Realization -> positions`` of (some of the atoms of) a molecule.

    This is what :class:`KNNDependency` queries with. Two queries are equal when they ask the same
    molecule (by identity) for the same atoms, which is what lets dependencies that query the same
    atoms share one search. The molecule is kept alive by the query, so the identity cannot be
    reused while the query exists.

    Parameters
    ----------
    mol : Mol
        The molecule to get the positions of.
    mask : array_like[bool], optional
        Only the atoms where this is true. All atoms by default.

    See Also
    --------
    KNNDependency : Queries with this.

    Examples
    --------
    >>> query = PositionQuery(ligand, mask=ligand.scoring_mask)
    >>> query(realization).shape  # the positions of the scored atoms
    """

    def __init__(self, mol: Mol, mask=None):
        self.mol = mol
        self.mask = None if mask is None else np.asarray(mask, dtype=bool)
        self._mask_key = None if self.mask is None else self.mask.tobytes()

    def __call__(self, realized: Realization) -> NDArray:
        positions = realized.positions(self.mol)
        return positions if self.mask is None else positions[..., self.mask, :]

    def __hash__(self):
        return hash((self.mol, self._mask_key))

    def __eq__(self, other):
        return (
            isinstance(other, PositionQuery)
            and other.mol is self.mol
            and other._mask_key == self._mask_key
        )


class PositionDependency(Dependency):
    """The atom positions of a molecule, for scoring functions that need them directly.

    Computed in numpy from the pose, once per molecule and call, however many terms ask. Shape
    ``(n_atoms, 3)``, or ``(n_poses, n_atoms, 3)`` for a batch.

    Parameters
    ----------
    mol : Mol
        The molecule to get the positions of.

    See Also
    --------
    Realization.positions : Where the positions come from.
    pyrite.Mol.pose_to_positions : Computes them.

    Examples
    --------
    >>> self.position_dep = PositionDependency(self.mol)  # in __init__
    >>> positions = computed[self.position_dep]  # in _score
    """

    def __init__(self, mol: Mol):
        self.mol = mol

    def compute(self, realized: Realization) -> NDArray:
        return realized.positions(self.mol)

    @classmethod
    def group_key(cls, dep):
        return dep.mol

    @classmethod
    def merge_group(cls, deps):
        return deps[0]


class RDKitDependency(Dependency):
    """A private RDKit copy of a molecule with the pose as its conformer.

    For scoring functions that need RDKit itself.

    The result is an :class:`rdkit.Chem.rdchem.Mol` with one conformer (the default one,
    ``confId=-1``): a list of them for a batch. It is shared by every term that depends on the
    same molecule, so a composite of several RDKit-based terms makes one copy per pose. It is a
    copy, so a term may do anything to it, and the scored :class:`~pyrite.Mol` is never touched.

    Parameters
    ----------
    mol : Mol
        The molecule to make the posed copy of.

    See Also
    --------
    pyrite.scoring._base._RDKitScoringFunction : The base class of RDKit-based terms.
    pyrite.Mol.to_rdkit : An RDKit copy with a pose, outside scoring.

    Examples
    --------
    >>> self.rdkit_dep = RDKitDependency(self.mol)  # in __init__
    >>> posed = computed[self.rdkit_dep]  # in _score: an RDKit molecule with the pose
    """

    def __init__(self, mol: Mol):
        self.mol = mol

    def compute(self, realized: Realization) -> Chem.Mol | list[Chem.Mol]:
        return realized.rdkit(self.mol)

    @classmethod
    def group_key(cls, dep):
        return dep.mol

    @classmethod
    def merge_group(cls, deps):
        return deps[0]


class KDTreeCache:  # pylint: disable=too-few-public-methods
    """
    Cache object holding KDTrees.

    Can be used to store KDTrees and access them globally.

    See Also
    --------
    KNNDependency : Builds its tree through this cache.

    Examples
    --------
    >>> tree = KDTreeCache.get_tree(hash(points.tobytes()), points)
    >>> distances, indices = tree.query(query_points, k=10)
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

        See Also
        --------
        scipy.spatial.KDTree : The tree.

        Examples
        --------
        >>> tree = KDTreeCache.get_tree(key, receptor.get_positions())
        """
        tree = cls._trees.get(key)
        if tree is None:
            tree = KDTree(point_cloud)
            cls._trees[key] = tree
        return tree


class KNNDependency(Dependency):
    """
    K-nearest neighbors dependency.

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
    point_cloud : NDArray
        The point cloud used to build the ``KDTree``.
    query_f : Callable[[Realization], NDArray]
        A callable that gets the points to query on the ``KDTree`` from a :class:`Realization`,
        with shape ``(..., n_points, 3)``. Use :class:`PositionQuery` for the positions of a
        molecule. This is combined with the point cloud to create the `group_key`, so it should
        compare equal for queries that can share a search.
    k : int
        The number of neighbors to retrieve.
    distance_upper_bound : float
        The upper bound of distance to consider when retrieving neighbors.

    See Also
    --------
    PositionQuery : The points to query, from a pose.
    pyrite.scoring.protein._KNNScoringFunction : The scoring functions built on it.

    Examples
    --------
    >>> query = PositionQuery(ligand, mask=ligand.scoring_mask)
    >>> nn_dep = KNNDependency(receptor.get_positions(), query, k=100, distance_upper_bound=8.0)
    >>> r, idx, mask = computed[nn_dep]  # in a scoring function's _score
    """

    def __init__(
        self,
        point_cloud: NDArray,
        query_f: Callable[[Realization], NDArray],
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
        self, realized: Realization
    ) -> tuple[
        NDArray,
        NDArray,
        NDArray,
    ]:
        """Execute the nearest neighbor search, for one pose or for a batch at once.

        All query points are searched in one call to the tree, for a batch of poses too. Neighbors
        beyond `distance_upper_bound` are not found: their index is the number of points in the
        tree, and the mask is false.

        Parameters
        ----------
        realized : Realization
            The pose(s) to query the positions of.

        Returns
        -------
        r : NDArray
            The distances to the nearest neighbors, shape ``(..., n_points, k)``, where ``...``
            is ``n_poses`` for a batch and nothing for one pose.
        idx : NDArray
            The indices of the nearest neighbors, same shape.
        mask : NDArray
            A boolean mask indicating which neighbors are valid, same shape.

        See Also
        --------
        narrow : Keep the k nearest within a term's cutoff.
        scipy.spatial.KDTree.query : The search.

        Examples
        --------
        >>> r, idx, mask = nn_dep.compute(Realization(pose, batched=False))
        >>> r.shape  # (n_query_points, k)
        """
        positions = self.querying(realized)
        r, idx = self.tree.query(
            positions.reshape(-1, 3),
            k=self.k,
            distance_upper_bound=self.distance_upper_bound,
        )
        # SciPy squeezes the neighbour axis for k == 1; always keep it as the last axis
        shape = (*positions.shape[:-1], self.k)
        r, idx = r.reshape(shape), idx.reshape(shape)
        return r, idx, (idx != self.tree.n)

    def narrow(self, computed):
        """Keep the `k` nearest neighbors, within `distance_upper_bound`, of a merged search.

        The merged search runs at the largest `k` and cutoff of its group. The neighbors are
        sorted nearest first, so the first `k` columns are kept, and neighbors beyond this
        dependency's cutoff are masked out.

        Parameters
        ----------
        computed : tuple[NDArray, NDArray, NDArray]
            The ``(r, idx, mask)`` of the merged search, with at least `k` neighbors.

        Returns
        -------
        tuple[NDArray, NDArray, NDArray]
            The ``(r, idx, mask)`` of this dependency.

        See Also
        --------
        merge_group : Makes the merged search.

        Examples
        --------
        >>> r, idx, mask = nn_dep.narrow(merged_result)
        """
        r, idx, mask = computed
        r, idx, mask = r[..., : self.k], idx[..., : self.k], mask[..., : self.k]
        mask = mask & (r < self.distance_upper_bound)
        return r, idx, mask

    @classmethod
    def group_key(cls, dep):
        """Return the group_key identifier of a dependency.

        The ``group_key`` of ``KNNDependency`` is a tuple containing the `tree_hash` and `querying`.

        Parameters
        ----------
        dep : Dependency
            The dependency to get the group key identifier for.


        Returns
        -------
        tuple

        See Also
        --------
        merge_group : Merges the dependencies with the same key.

        Examples
        --------
        >>> KNNDependency.group_key(nn_dep) == KNNDependency.group_key(other_dep)
        True
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

        See Also
        --------
        narrow : Gives every dependency its own part back.

        Examples
        --------
        >>> merged = KNNDependency.merge_group([narrow_dep, wide_dep])
        >>> merged.k == max(narrow_dep.k, wide_dep.k)
        True
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
