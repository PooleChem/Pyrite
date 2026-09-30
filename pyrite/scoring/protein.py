from abc import ABC
from enum import IntEnum
from typing import Any

import numpy as np
from numba import njit
from numpy._typing import NDArray

from .._common import Mol
from ..atom_consts import AtomType, vina_atom_consts
from ._base import ScoringFunction
from .dependencies import Dependency, KNNDependency


@njit
def _gaussian_kernel(x, w):
    return np.exp(-((x / w) ** 2))


@njit
def _slope_step_kernel(dist, good, bad):
    # Numba does not support 2D boolean fancy indexing, so we flatten and loop.
    flat = dist.ravel()
    n = flat.shape[0]
    out = np.empty(n)
    slope = good - bad
    for k in range(n):
        d = flat[k]
        if d >= bad:
            out[k] = 0.0
        elif d <= good:
            out[k] = 1.0
        else:
            out[k] = (d - bad) / slope
    return out.reshape(dist.shape)


@njit
def _lj_kernel(r, optimal_distance, i_exp, j_exp, smoothing, cap, depth, mask):
    # c_i / c_j are 2D element-wise ops — supported by Numba.
    c_i = (optimal_distance ** i_exp) * depth * j_exp / (i_exp - j_exp)
    c_j = (optimal_distance ** j_exp) * depth * i_exp / (j_exp - i_exp)

    # Accumulate in a loop to avoid 2D boolean fancy indexing.
    total = 0.0
    for row in range(r.shape[0]):
        for col in range(r.shape[1]):
            if not mask[row, col]:
                continue
            opt = optimal_distance[row, col]
            rval = r[row, col]
            if rval > opt + smoothing:
                r2 = rval - smoothing
            elif rval < opt - smoothing:
                r2 = rval + smoothing
            else:
                r2 = opt
            total += min(cap, c_i[row, col] / r2 ** i_exp + c_j[row, col] / r2 ** j_exp)
    return total


@njit
def _four_piece_kernel(r, a, b, c, d, e, f):
    res = np.zeros(len(r))
    for k in range(len(r)):
        ri = r[k]
        if ri < a:
            res[k] = f * (a - ri) / a
        elif ri < b:
            res[k] = e * (ri - a) / (b - a)
        elif ri < c:
            res[k] = e
        elif ri < d:
            res[k] = e * (d - ri) / (d - c)
    return res


@njit
def _two_piece_kernel(r, a, b, c, d):
    res = np.zeros(len(r))
    for k in range(len(r)):
        ri = r[k]
        if ri < a:
            res[k] = ri * (c - d) / a + d
        elif ri <= b:
            res[k] = -c * (ri - a) / (b - a) + c
    return res


class _KNNScoringFunction(ScoringFunction, ABC):
    r"""
    ⚙️ — K-Nearest neighbor distance scoring function.

    This abstract scoring function can be used to implement scoring functions that make use of the
    distance of `probe_mol` atoms to the closest `k` protein atoms.

    .. note::
        This is an abstract base class and should thus be subclassed. Please refer to TODO
         for more information on how to do this.


    **Speed**: ⚙️, depends on implementation.

    Parameters
    ----------
    probe_mol : Mol
        The probe molecule to be used for the calculation.
    fixed_mol : Mol
        The fixed molecule to be used for the calculation.
    cutoff : float, default 8.0
        NOT WORKING. Maximum distance to consider in the nearest neighbor search.
    k : int, default 400
        The number of neighbors to consider.

    See Also
    --------
    _ChargeScoringFunction
        Abstract charge-dependent scoring function.

    """

    def __init__(
        self,
        probe_mol: Mol,
        fixed_mol: Mol,
        cutoff: float = 8.0,
        k: int = 400,
    ):
        self.probe_mol = probe_mol
        self.fixed_mol = fixed_mol

        self.probe_mask = probe_mol.scoring_mask
        self.fixed_mask = fixed_mol.scoring_mask

        self.cutoff = cutoff
        self.k = k
        self.offset = 0.0

        self.__init_radii()

        self.nn_dep = KNNDependency(
            self.fixed_mol.positions[self.fixed_mask],
            self.probe_mol.get_positions,
            self.k,
            self.cutoff,
        )
        self._tree_n = len(self.fixed_mol.positions)

    def __init_radii(self):
        max_type = max(e.value for e in AtomType)
        self.xs_radii = np.empty(max_type + 1, dtype=float)
        for _, t in vina_atom_consts.items():
            self.xs_radii[t.type.value] = t.xs_radius

        self.probe_radii = self.xs_radii[self.probe_mol.atom_types[self.probe_mask]]
        # Precomputed once: fixed_mask/fixed_mol.atom_types never change after construction,
        # so re-filtering them on every _optimal_distance call (once per KNN term, per
        # pose) was pure waste.
        self._fixed_radii_masked = self.xs_radii[self.fixed_mol.atom_types[self.fixed_mask]]

    def get_dependencies(self) -> list[Dependency]:
        return [self.nn_dep]

    def _optimal_distance(self, idx, mask, radii, offset: float = 0.0):
        safe_idx = np.where(mask, idx, 0)
        fixed_radii = self._fixed_radii_masked[safe_idx]

        return radii + fixed_radii + offset

    def _kernel(self, dist: NDArray) -> NDArray:
        """The elementwise score for each neighbor, as a function of
        ``(distance - optimal_distance - offset)``. The only thing a
        concrete ``_KNNScoringFunction`` needs to implement.
        """
        raise NotImplementedError

    def _mask(self, idx, mask):
        """Optional extra neighbor-pairing constraint (e.g. hydrophobic/hbond
        pairing), on top of plain neighbor validity. Default: no extra
        restriction.
        """
        return mask

    def _mask_field(self, idx, mask, atom_type):
        return mask

    def _score(self, conf_id, computed) -> float:
        r, idx, mask = computed[self.nn_dep]
        r = r[self.probe_mask]
        idx = idx[self.probe_mask]
        mask = mask[self.probe_mask]

        dist = r - self._optimal_distance(idx, mask, self.probe_radii[:, None], self.offset)
        s = self._kernel(dist)

        s[~self._mask(idx, mask)] = 0.0
        return np.sum(s)

    def _score_field(self, r, idx, atom_type: AtomType, mask=None) -> NDArray[np.float64]:
        # mask defaults to None (recomputed from idx) for the grid-construction
        # caller (GridScore builds r/idx directly from a raw KDTree query, with no
        # pre-existing narrowed mask to inherit). _batch_scores (below) passes its
        # own already-narrowed mask explicitly — recomputing from idx alone there
        # would silently undo KNNDependency.narrow()'s per-term cutoff masking,
        # since narrow() only clears the boolean mask, not idx/r themselves.
        if mask is None:
            mask = idx != self.nn_dep.tree.n

        # [..., None] adds the trailing neighbor axis so this broadcasts correctly whether
        # atom_type is a scalar (single hypothetical type, e.g. a grid sweep) or an array
        # matching the atoms axis (a real ligand's per-atom types, e.g. a batch of poses) —
        # a no-op for the scalar case, required for the array case.
        radii = np.asarray(self.xs_radii[atom_type])[..., None]
        dist = r - self._optimal_distance(idx, mask, radii, self.offset)
        s = self._kernel(dist)

        s[~self._mask_field(idx, mask, atom_type)] = 0.0
        return s.sum(axis=-1)

    def _batch_scores(self, conf_ids, computed_batch) -> NDArray[np.float64]:
        r, idx, mask = computed_batch[self.nn_dep]
        r = r[:, self.probe_mask]
        idx = idx[:, self.probe_mask]
        mask = mask[:, self.probe_mask]

        # the real ligand's own per-atom types — _score_field already handles an
        # array atom_type (verified), summing here over the remaining atoms axis
        # (it already summed over neighbors) gives one score per conformer.
        atom_types = self.probe_mol.atom_types[self.probe_mask]
        return self._score_field(r, idx, atom_types, mask=mask).sum(axis=-1)


class Gaussian(_KNNScoringFunction):
    r"""
    🚶 — A Gaussian function of the distance.

    .. math::
        score = \exp(-\frac{distance - (optimal\_distance + offset)}{width}^2)

    Here `optimal_distance` is the sum of the ``xs_radii`` of the two atoms.

    This results in the following function, where in this example the `optimal_distance` is 4.0,
    the `offset` is 0.0, and the `width` is 0.5.

    .. plot::
       :width: 80%
       :alt: Gaussian example

       import numpy as np
       import matplotlib.pyplot as plt

       optimal = 4.0
       offset = 0.0
       width = 0.5

       x = np.linspace(0.0001, 6, 400)
       y = np.exp(-((((x - (optimal + offset)) / width)) ** 2))

       plt.figure()
       plt.grid(visible=True)
       plt.plot(x, y, linewidth=2)
       plt.xlabel("Distance")
       plt.ylabel("Score")

    This score is calculated for every `probe_mol` atom to all its neighbors
    (as defined by `cutoff` and `k`), and then summed.

    **Speed**: 🐢–🚗, depending on `cutoff` and `k`.

    Parameters
    ----------
    probe_mol : Mol
        The probe molecule to be used for the calculation.
    fixed_mol : Mol
        The fixed molecule to be used for the calculation.
    offset : float, default 0.0
        The offset from `optimal_distance` that is considered as ideal.
    width : float, default 0.5
        The width of the Gaussian.
    cutoff : float, default 8.0
        NOT WORKING. Maximum distance to consider in the nearest neighbor search.
    k : int, default 400
        The number of neighbors to consider. It is necessary to increase this number if a wider or
        higher offset Gaussian is used.

    """

    def __init__(
        self,
        probe_mol: Mol,
        fixed_mol: Mol,
        offset: float = 0.0,
        width: float = 0.5,
        cutoff: float = None,
        k: int = 400,
    ):
        super().__init__(
            probe_mol,
            fixed_mol,
            cutoff or 4 + offset + np.e * width,
            k,
        )
        self.offset = offset
        self.width = width

    def _kernel(self, dist):
        return _gaussian_kernel(dist, self.width)


class Repulsion(_KNNScoringFunction):
    r"""
    🚲 — Decreases exponentially with the distance.

    .. math::
        score = min(distance - (optimal\_distance + offset), 0.0)^2

    Here `optimal_distance` is the sum of the ``xs_radii`` of the two atoms.

    This results in the following function, where in this example the `optimal_distance` is 4.0
    and the `offset` is 0.0.

    .. plot::
       :width: 80%
       :alt: Repulsion example

       import numpy as np
       import matplotlib.pyplot as plt

       optimal = 4.0
       offset = 0.0

       x = np.linspace(0.0001, 5, 400)
       y = np.minimum(x - (optimal + offset), 0.0)**2

       plt.figure()
       plt.grid(visible=True)
       plt.plot(x, y, linewidth=2)
       plt.xlabel("Distance")
       plt.ylabel("Score")

    This score is calculated for every `probe_mol` atom to all its neighbors
    (as defined by `cutoff` and `k`), and then summed.

    **Speed**:🚲

    Parameters
    ----------
    probe_mol : Mol
        The probe molecule to be used for the calculation.
    fixed_mol : Mol
        The fixed molecule to be used for the calculation.
    offset : float, default 0.0
        The offset that is added to `optimal_distance`.
    cutoff : float, default 8.0
        NOT WORKING. Maximum distance to consider in the nearest neighbor search.
    k : int, default 400
        The number of neighbors to consider.
    """

    def __init__(
        self,
        probe_mol: Mol,
        fixed_mol: Mol,
        offset: float = 0.0,
        cutoff: float = None,
        k: int = 400,
    ):
        super().__init__(
            probe_mol, fixed_mol, cutoff or 4 + offset, k
        )
        self.offset = offset

    def _kernel(self, dist):
        d = np.minimum(dist, 0.0)
        return d * d


class _SlopeStep(_KNNScoringFunction):
    r"""
    ⚙️ — Slope Step scoring of distance.

    This abstract scoring function can be used to implement scoring functions that make use of a
    slope step function:

    .. plot::
       :width: 80%
       :alt: Slope Step example

       import numpy as np
       import matplotlib.pyplot as plt

       good = 0.5
       bad = 1.5

       x = np.linspace(0, 3, 400)

       y = (x - bad) / (good - bad)
       y[x >= bad] = 0.0
       y[x <= good] = 1.0

       plt.figure()
       plt.grid(visible=True)
       plt.plot(x, y, linewidth=2)
       plt.xlabel("Distance")
       plt.ylabel("Score")


    .. note::
        This is an abstract base class and should thus be subclassed. Please refer to TODO
         for more information on how to do this.


    **Speed**: ⚙️, depends on implementation.

    Parameters
    ----------
    probe_mol : Mol
        The probe molecule to be used for the calculation.
    fixed_mol : Mol
        The fixed molecule to be used for the calculation.
    good : float, default 0.5
        The `good` distance. Distance values lower than this value are scored :math:`1`.
    bad : float, default 1.5
        The `bad` distance. Distance values higher than this value are scored :math:`0`.
    cutoff : float, default 8.0
        NOT WORKING. Maximum distance to consider in the nearest neighbor search.
    k : int, default 400
        The number of neighbors to consider.

    See Also
    --------
    _ChargeScoringFunction
        Abstract charge-dependent scoring function.

    """

    def __init__(
        self,
        probe_mol: Mol,
        fixed_mol: Mol,
        good: float = 0.5,
        bad: float = 1.5,
        cutoff: float = None,
        k: int = 400,
    ):
        assert good < bad, "Bad distance <= good distance not implemented."
        # `bad` is an offset from optimal_distance (~4 Å for a typical atom pair),
        # not an absolute search radius — cutoff needs that base distance added,
        # same pattern as Gaussian's `4 + offset + ...`/Repulsion's `4 + offset`.
        super().__init__(probe_mol, fixed_mol, cutoff or (4 + bad), k)
        self.good = good
        self.bad = bad

    def _kernel(self, dist):
        return _slope_step_kernel(dist, self.good, self.bad)


class Hydrophobic(_SlopeStep):
    r"""
    🚲 — Slope-step between hydrophobic atoms.

    .. math::
        delta = distance - optimal\_distance

    .. math::
        score =
        \begin{cases}
        1.0, & delta \le good,\\[4pt]
        \frac{delta - bad}{good - bad}, & good \le delta \le bad,\\[4pt]
        0.0, & bad \le delta.
        \end{cases}

    Here `optimal_distance` is the sum of the ``xs_radii`` of the two atoms.

    This results in the following function, where in this example the `optimal_distance` is 4.0,
    `good` is 0.5 and `bad` is 1.5.

    .. plot::
       :width: 80%
       :alt: Hydrophobic example

       import numpy as np
       import matplotlib.pyplot as plt

       optimal = 4.0
       good = 0.5
       bad = 1.5

       x = np.linspace(2, 8, 400)
       delta = x - optimal

       y = (delta - bad) / (good - bad)

       y[delta <= good] = 1.0
       y[delta >= bad] = 0.0


       plt.figure()
       plt.ylim((-0.5, 1.5))
       plt.grid(visible=True)
       plt.plot(x, y, linewidth=2)
       plt.xlabel("Distance")
       plt.ylabel("Score")

    This score is calculated for every hydrophobic `probe_mol` atom to all its hydrophobic neighbors
    (as defined by `cutoff` and `k`), and then summed. The hydrophobic neighbors are determined
    after the nearest neighbor search.

    **Speed**:🚲

    Parameters
    ----------
    probe_mol : Mol
        The probe molecule to be used for the calculation.
    fixed_mol : Mol
        The fixed molecule to be used for the calculation.
    good : float, default 0.5
        The start of the `slope-step`.
    bad : float, default 1.5
        The end of the `slope-step`.
    cutoff : float, default 8.0
        NOT WORKING. Maximum distance to consider in the nearest neighbor search.
    k : int, default 400
        The number of neighbors to consider.

    See Also
    --------
    NonHydrophobic
        For hydrophilic interactions.

    """

    def __init__(
        self,
        probe_mol: Mol,
        fixed_mol: Mol,
        good: float = 0.5,
        bad: float = 1.5,
        cutoff: float = None,
        k: int = 400,
    ):
        super().__init__(
            probe_mol, fixed_mol, good, bad, cutoff, k
        )

        self.__init_hydrophobic()

    def __init_hydrophobic(self):
        max_type = max(e.value for e in AtomType)
        self.xs_hydrophobic = np.empty(max_type + 1, dtype=bool)
        for _, t in vina_atom_consts.items():
            self.xs_hydrophobic[t.type.value] = t.xs_hydrophobe

        self.probe_mol_hydrophobic = self.xs_hydrophobic[
            self.probe_mol.atom_types[self.probe_mask]
        ]
        # Precomputed once — see _KNNScoringFunction.__init_radii for why.
        self._fixed_hydrophobic_masked = self.xs_hydrophobic[
            self.fixed_mol._atom_types[self.fixed_mask]
        ]

    def _mask(self, idx, neighbor_mask):
        safe_idx = np.where(neighbor_mask, idx, 0)
        fixed_mol_hydrophobic = self._fixed_hydrophobic_masked[safe_idx]
        return (
            self.probe_mol_hydrophobic[:, None]
            & fixed_mol_hydrophobic[:, :]
            & neighbor_mask
        )

    def _mask_field(self, idx, mask, atom_type):
        fixed_hydrophobic = self._fixed_hydrophobic_masked[np.where(mask, idx, 0)]
        hydrophobic = np.asarray(self.xs_hydrophobic[atom_type])[..., None]
        return hydrophobic & fixed_hydrophobic[:, :] & mask


class NonHydrophobic(Hydrophobic):
    r"""
    🚲 — Slope-step between non-hydrophobic bonds.

    .. math::
        delta = distance - optimal\_distance

    .. math::
        score =
        \begin{cases}
        1.0, & delta \le good,\\[4pt]
        \frac{delta - bad}{good - bad}, & good \le delta \le bad,\\[4pt]
        0.0, & bad \le delta.
        \end{cases}

    Here `optimal_distance` is the sum of the ``xs_radii`` of the two atoms.

    This results in the following function, where in this example the `optimal_distance` is 4.0,
    `good` is 0.5 and `bad` is 1.5.

    .. plot::
       :width: 80%
       :alt: Nonhydrophobic example

       import numpy as np
       import matplotlib.pyplot as plt

       optimal = 4.0
       good = 0.5
       bad = 1.5

       x = np.linspace(2, 8, 400)
       delta = x - optimal

       y = (delta - bad) / (good - bad)

       y[delta <= good] = 1.0
       y[delta >= bad] = 0.0


       plt.figure()
       plt.ylim((-0.5, 1.5))
       plt.grid(visible=True)
       plt.plot(x, y, linewidth=2)
       plt.xlabel("Distance")
       plt.ylabel("Score")

    This score is calculated for every nonhydrophobic `probe_mol` atom to all its
    nonhydrophobic neighbors (as defined by `cutoff` and `k`), and then summed.
    The nonhydrophobic neighbors are determined after the nearest neighbor search.

    **Speed**:🚲

    Parameters
    ----------
    probe_mol : Mol
        The probe molecule to be used for the calculation.
    fixed_mol : Mol
        The fixed molecule to be used for the calculation.
    good : float, default 0.5
        The start of the `slope-step`.
    bad : float, default 1.5
        The end of the `slope-step`.
    cutoff : float, default 8.0
        NOT WORKING. Maximum distance to consider in the nearest neighbor search.
    k : int, default 400
        The number of neighbors to consider.

    See Also
    --------
    Hydrophobic
        For hydrophobic interactions.

    """

    def _mask(self, idx, neighbor_mask):
        safe_idx = np.where(neighbor_mask, idx, 0)
        fixed_mol_hydrophobic = self._fixed_hydrophobic_masked[safe_idx]
        return ~self.probe_mol_hydrophobic[:, None] & ~fixed_mol_hydrophobic & neighbor_mask

    def _mask_field(self, idx, mask, atom_type):
        fixed_hydrophobic = self._fixed_hydrophobic_masked[np.where(mask, idx, 0)]
        hydrophobic = np.asarray(self.xs_hydrophobic[atom_type])[..., None]
        return ~hydrophobic & ~fixed_hydrophobic[:, :] & mask


class NonDirHBond(_SlopeStep):
    r"""
    🚲 — Slope-step within hydrogen bonds.

    .. math::
        delta = distance - optimal\_distance

    .. math::
        score =
        \begin{cases}
        1.0, & delta \le good,\\[4pt]
        \frac{delta - bad}{good - bad}, & good \le delta \le bad,\\[4pt]
        0.0, & bad \le delta.
        \end{cases}

    Here `optimal_distance` is the sum of the ``xs_radii`` of the two atoms.

    This results in the following function, where in this example the `optimal_distance` is 4.0,
    `good` is -0.7 and `bad` is 0.

    .. plot::
       :width: 80%
       :alt: HBond example

       import numpy as np
       import matplotlib.pyplot as plt

       optimal = 4.0
       good = -0.7
       bad = 0.0

       x = np.linspace(0.0001, 6, 400)
       delta = x - optimal

       y = (delta - bad) / (good - bad)

       y[delta <= good] = 1.0
       y[delta >= bad] = 0.0


       plt.figure()
       plt.ylim((-0.5, 1.5))
       plt.grid(visible=True)
       plt.plot(x, y, linewidth=2)
       plt.xlabel("Distance")
       plt.ylabel("Score")

    This score is calculated for every hbond-acceptor `probe_mol` atom to all its hbond-donor neighbors
    and from every hbond-donor `probe_mol` atom to all its hbond-acceptor neighbors
    (as defined by `cutoff` and `k`), and then summed. The hbond neighbors are determined
    after the nearest neighbor search.

    **Speed**:🚲

    Parameters
    ----------
    probe_mol : Mol
        The probe molecule to be used for the calculation.
    fixed_mol : Mol
        The fixed molecule to be used for the calculation.
    good : float, default -0.7
        The start of the `slope-step`.
    bad : float, default 0
        The end of the `slope-step`.
    cutoff : float, default 8.0
        NOT WORKING. Maximum distance to consider in the nearest neighbor search.
    k : int, default 400
        The number of neighbors to consider.
    See Also
    --------
    NonDirHBondLJ
        For a hydrogen bond implementation using a Lennard-Jones potential.

    """

    def __init__(
        self,
        probe_mol: Mol,
        fixed_mol: Mol,
        good: float = -0.7,
        bad: float = 0,
        cutoff: float = None,
        k: int = 400,
    ):
        super().__init__(
            probe_mol, fixed_mol, good, bad, cutoff, k
        )

        self.__init_hbond_possible()

    def __init_hbond_possible(self):
        max_type = max(e.value for e in AtomType)
        self.xs_acceptor = np.empty(max_type + 1, dtype=bool)
        self.xs_donor = np.empty(max_type + 1, dtype=bool)
        for _, t in vina_atom_consts.items():
            self.xs_acceptor[t.type.value] = t.xs_acceptor
            self.xs_donor[t.type.value] = t.xs_donor

        self.probe_mol_acceptor = self.xs_acceptor[self.probe_mol.atom_types[self.probe_mask]]
        self.probe_mol_donor = self.xs_donor[self.probe_mol.atom_types[self.probe_mask]]
        # Precomputed once — see _KNNScoringFunction.__init_radii for why.
        self._fixed_acceptor_masked = self.xs_acceptor[self.fixed_mol._atom_types[self.fixed_mask]]
        self._fixed_donor_masked = self.xs_donor[self.fixed_mol._atom_types[self.fixed_mask]]

    def _mask(self, idx, neighbor_mask):
        safe_idx = np.where(neighbor_mask, idx, 0)

        fixed_mol_acceptor = self._fixed_acceptor_masked[safe_idx]
        fixed_mol_donor = self._fixed_donor_masked[safe_idx]
        return (
            (self.probe_mol_donor[:, None] & fixed_mol_acceptor)
            | (self.probe_mol_acceptor[:, None] & fixed_mol_donor)
        ) & neighbor_mask

    def _mask_field(self, idx, mask, atom_type):
        safe_idx = np.where(mask, idx, 0)
        fixed_acceptor = self._fixed_acceptor_masked[safe_idx]
        fixed_donor = self._fixed_donor_masked[safe_idx]
        donor = np.asarray(self.xs_donor[atom_type])[..., None]
        acceptor = np.asarray(self.xs_acceptor[atom_type])[..., None]
        return (
            (donor & fixed_acceptor)
            | (acceptor & fixed_donor)
        ) & mask


class LJ(_KNNScoringFunction):
    r"""
    🐢 — Lennard-Jones potential.

    Scores distance based on a Lennard-Jones potential.


    Firstly, two Van Der Waals coefficients are determined:

    .. math::
        c_i = \frac{{(optimal\_distance + offset)}^{i} \times depth \times j}{i - j}

        c_j = \frac{{(optimal\_distance + offset)}^{j} \times depth \times i}{j - i}

    Here `optimal_distance` is the sum of the ``xs_radii`` of the two atoms.

    Next, an optional smoothing step is applied to the distance:

    .. math::
        r =
        \begin{cases}
        r - smoothing, & r > (optimal\_distance + smoothing),\\[4pt]
        r + smoothing, & r < (optimal\_distance - smoothing),\\[4pt]
        optimal\_distance, & otherwise.
        \end{cases}

    Finally, the score is calculated as:

    .. math::
        \min(cap, \frac{c_i}{r^i} + \frac{c_j}{r^j})


    This results in the following function, where in this example the `optimal_distance` is 4.0,
    `i` is 10 and `j` is 12, smoothing is 0.0, `cap` is 10.0 and `depth` is 1.0.

    .. plot::
       :width: 80%
       :alt: Lennard-Jones example

       import numpy as np
       import matplotlib.pyplot as plt

       optimal = 4.0
       i = 10
       j = 12
       cap = 10

       c_i = (optimal**i) * j / (i - j)
       c_j = (optimal**j) * i / (j - i)

       x = np.linspace(0.0001, 12, 400)

       r_i = x**i
       r_j = x**j

       y = np.minimum(cap, c_i / r_i + c_j / r_j)

       plt.figure()
       plt.grid(visible=True)
       plt.plot(x, y, linewidth=2)
       plt.xlabel("Distance")
       plt.ylabel("Score")

    This score is calculated for every `probe_mol` atom to all its neighbors
    (as defined by `cutoff` and `k`), and then summed.

    **Speed**: 🐢–🚶, depending on `cutoff` and `k`.

    Parameters
    ----------
    probe_mol : Mol
        The probe molecule to be used for the calculation.
    fixed_mol : Mol
        The fixed molecule to be used for the calculation.
    i : int, default 10
        The first exponent of the Lennard-Jones potential.
    j : int, default 12
        The second exponent of the Lennard-Jones potential.
    smoothing : float, optional
        An optional smoothing step to apply to the distance. Values within `smoothing` of the
        optimal will be set to the optimal distance.
    offset : float, default 0.0
        The offset from the optimal distance.
    cap : float, default 100.0
        The maximum score for a single atom-atom interaction.
    depth : float, default 1.0
        The depth of the LJ-potential minimum.
    cutoff : float, default 8.0
        Maximum distance to consider in the nearest neighbor search.
    k : int, default 400
        The number of neighbors to consider.

    """

    def __init__(
        self,
        probe_mol: Mol,
        fixed_mol: Mol,
        i: int = 10,
        j: int = 12,
        smoothing: float = 0,
        offset: float = 0.0,
        cap: float = 100.0,
        depth: float = 1.0,
        cutoff: float = None,
        k: int = 400,
    ):
        super().__init__(
            probe_mol,
            fixed_mol,
            cutoff or 8 + offset,
            k,
        )

        self._i = i
        self._j = j
        self._smoothing = smoothing
        self._offset = offset
        self._cap = cap
        self._depth = depth

    def _score(self, conf_id, computed) -> float:
        r, idx, neighbor_mask = computed[self.nn_dep]
        r = r[self.probe_mask]
        idx = idx[self.probe_mask]
        neighbor_mask = neighbor_mask[self.probe_mask]

        optimal_distance = self._optimal_distance(idx, neighbor_mask, self.probe_radii[:, None], self._offset)
        mask = self._mask(idx, neighbor_mask)

        return _lj_kernel(
            r, optimal_distance,
            self._i, self._j, self._smoothing, self._cap, self._depth,
            mask,
        )

    # LJ's kernel needs r and optimal_distance separately (not just their difference) and
    # masks+accumulates inside the numba loop — it doesn't fit _KNNScoringFunction's
    # _kernel(dist) shape, so it never implements _kernel. Without this, LJ would silently
    # *inherit* _KNNScoringFunction's _kernel-based _score_field/_batch_scores and crash with
    # NotImplementedError only when actually called. _score_field = None makes GridScore's
    # support check correctly reject it; _batch_scores falls back to ScoringFunction's
    # always-correct generic default instead of the broken kernel-based one.
    _score_field = None
    _batch_scores = ScoringFunction._batch_scores


class VDW(LJ):
    r"""
    🐢 — Van Der Waals force based on Lennard-Jones potential.

    Scores distance based on a Lennard-Jones potential with a depth of 1.


    Firstly, two Van Der Waals coefficients are determined:

    .. math::
        c_i = \frac{{optimal\_distance}^{i} \times j}{i - j}

        c_j = \frac{{optimal\_distance}^{j} \times i}{j - i}

    Here `optimal_distance` is the sum of the ``xs_radii`` of the two atoms.

    Next, an optional smoothing step is applied to the distance:

    .. math::
        r =
        \begin{cases}
        r - smoothing, & r > (optimal\_distance + smoothing),\\[4pt]
        r + smoothing, & r < (optimal\_distance - smoothing),\\[4pt]
        optimal\_distance, & otherwise.
        \end{cases}

    Finally, the score is calculated as:

    .. math::
        \min(cap, \frac{c_i}{r^i} + \frac{c_j}{r^j})


    This results in the following function, where in this example the `optimal_distance` is 4.0,
    `i` is 4 and `j` is 8, smoothing is 1.0, and `cap` is 10.0.

    .. plot::
       :width: 80%
       :alt: Van Der Waals example

       import numpy as np
       import matplotlib.pyplot as plt

       optimal = 4.0
       i = 4
       j = 8
       cap = 10

       c_i = (optimal**i) * j / (i - j)
       c_j = (optimal**j) * i / (j - i)

       x = np.linspace(0.0001, 12, 400)

       x2 = np.ones(x.shape) * optimal
       x2[x > (optimal + 1)] = x[x > (optimal + 1)] - 1
       x2[x < (optimal - 1)] = x[x < (optimal - 1)] + 1

       r_i = x2**i
       r_j = x2**j

       y = np.minimum(cap, c_i / r_i + c_j / r_j)

       plt.figure()
       plt.grid(visible=True)
       plt.plot(x, y, linewidth=2)
       plt.xlabel("Distance")
       plt.ylabel("Score")

    This score is calculated for every `probe_mol` atom to all its neighbors
    (as defined by `cutoff` and `k`), and then summed.

    **Speed**: 🐢–🚶

    Parameters
    ----------
    probe_mol : Mol
        The probe molecule to be used for the calculation.
    fixed_mol : Mol
        The fixed molecule to be used for the calculation.
    i : int, default 4
        The first exponent of the Lennard-Jones potential.
    j : int, default 8
        The second exponent of the Lennard-Jones potential.
    smoothing : float, optional
        An optional smoothing step to apply to the distance. Values within `smoothing` of the
        optimal will be set to the optimal distance.
    cap : float, default 100.0
        The maximum score for a single atom-atom interaction.
    cutoff : float, default 8.0
        NOT WORKING. Maximum distance to consider in the nearest neighbor search.
    k : int, default 400
        The number of neighbors to consider.
    See Also
    --------
    LJ
        For the Lennard-Jones potential.

    """

    def __init__(
        self,
        probe_mol: Mol,
        fixed_mol: Mol,
        i: int = 4,
        j: int = 8,
        smoothing: float = 0,
        cap: float = 100.0,
        cutoff: float = None,
        k: int = 400,
    ):
        super().__init__(
            probe_mol,
            fixed_mol,
            i=i,
            j=j,
            smoothing=smoothing,
            offset=0,
            cap=cap,
            cutoff=cutoff,
            k=k,
        )


class NonDirHBondLJ(LJ):
    r"""
    🐢 — Hydrogen bonding force based on Lennard-Jones potential.

    Scores distance between hydrogen acceptors and donors based on a Lennard-Jones 10-12 potential
    with a depth of 5.


    Firstly, two Van Der Waals coefficients are determined:

    .. math::
        c_i = \frac{{(optimal\_distance + offset)}^{i} \times 5 \times j}{i - j}

        c_j = \frac{{(optimal\_distance + offset)}^{j} \times 5 \times i}{j - i}

    Here `optimal_distance` is the sum of the ``xs_radii`` of the two atoms.

    Finally, the score is calculated as:

    .. math::
        \min(cap, \frac{c_i}{r^i} + \frac{c_j}{r^j})


    This results in the following function, where in this example the `optimal_distance` is 4.0,
    `offset` is -0.7, and `cap` is 10.0.

    .. plot::
       :width: 80%
       :alt: Van Der Waals example

       import numpy as np
       import matplotlib.pyplot as plt

       optimal = 4.0
       i = 10
       j = 12
       cap = 10

       c_i = ((optimal - 0.7)**i) * 5 * j / (i - j)
       c_j = ((optimal - 0.7)**j) * 5 * i / (j - i)

       x = np.linspace(0.0001, 12, 400)

       r_i = x**i
       r_j = x**j

       y = np.minimum(cap, c_i / r_i + c_j / r_j)

       plt.figure()
       plt.grid(visible=True)
       plt.plot(x, y, linewidth=2)
       plt.xlabel("Distance")
       plt.ylabel("Score")

    This score is calculated for every hbond-acceptor `probe_mol` atom to all its hbond-donor neighbors
    and from every hbond-donor `probe_mol` atom to all its hbond-acceptor neighbors
    (as defined by `cutoff` and `k`), and then summed. The hbond neighbors are determined
    after the nearest neighbor search.

    **Speed**: 🐢–🚶

    Parameters
    ----------
    probe_mol : Mol
        The probe molecule to be used for the calculation.
    fixed_mol : Mol
        The fixed molecule to be used for the calculation.
    offset : float, default -0.7
        The offset from the optimal distance.
    cap : float, default 100.0
        The maximum score for a single atom-atom interaction.
    cutoff : float, default 8.0
        NOT WORKING. Maximum distance to consider in the nearest neighbor search.
    k : int, default 400
        The number of neighbors to consider.
    See Also
    --------
    NonDirHBond
        For a hydrogen bond implementation using a slope-step.

    """

    def __init__(
        self,
        probe_mol: Mol,
        fixed_mol: Mol,
        offset: float = -0.7,
        cap: float = 100.0,
        cutoff: float = None,
        k: int = 400,
    ):
        super().__init__(
            probe_mol,
            fixed_mol,
            i=10,
            j=12,
            smoothing=0,
            offset=offset,
            cap=cap,
            depth=5.0,
            cutoff=cutoff,
            k=k,
        )
        self.__init_hbond_possible()

    def __init_hbond_possible(self):
        max_type = max(e.value for e in AtomType)
        self.xs_acceptor = np.empty(max_type + 1, dtype=bool)
        self.xs_donor = np.empty(max_type + 1, dtype=bool)
        for _, t in vina_atom_consts.items():
            self.xs_acceptor[t.type.value] = t.xs_acceptor
            self.xs_donor[t.type.value] = t.xs_donor

        self.probe_mol_acceptor = self.xs_acceptor[self.probe_mol.atom_types[self.probe_mask]]
        self.probe_mol_donor = self.xs_donor[self.probe_mol.atom_types[self.probe_mask]]
        # Precomputed once — see _KNNScoringFunction.__init_radii for why.
        self._fixed_acceptor_masked = self.xs_acceptor[self.fixed_mol._atom_types[self.fixed_mask]]
        self._fixed_donor_masked = self.xs_donor[self.fixed_mol._atom_types[self.fixed_mask]]

    def _mask(self, idx, neighbor_mask):
        safe_idx = np.where(neighbor_mask, idx, 0)

        fixed_mol_acceptor = self._fixed_acceptor_masked[safe_idx]
        fixed_mol_donor = self._fixed_donor_masked[safe_idx]
        return (
            (self.probe_mol_donor[:, None] & fixed_mol_acceptor)
            | (self.probe_mol_acceptor[:, None] & fixed_mol_donor)
        ) & neighbor_mask


class _ChargeScoringFunction(_KNNScoringFunction, ABC):
    r"""
    ⚙️ — Gasteiger-charge based scoring

    This abstract scoring function can be used to implement scoring functions that make use of the
    distance of `probe_mol` atoms to the closest `k` protein atoms, and the charge of these atoms.

    These charges are calculated using the Gasteiger [1]_ method, using
    :func:`~rdkit.Chem.rdPartialCharges.ComputeGasteigerCharges`.

    .. note::
        This is an abstract base class and should thus be subclassed. Please refer to TODO
         for more information on how to do this.


    **Speed**: ⚙️, depends on implementation.

    Parameters
    ----------
    probe_mol : Mol
        The probe molecule to be used for the calculation.
    fixed_mol : Mol
        The fixed molecule to be used for the calculation.
    cutoff : float, default 8.0
        NOT WORKING. Maximum distance to consider in the nearest neighbor search.
    k : int, default 400
        The number of neighbors to consider.

    See Also
    --------
    ElectroStatic
        Simple charge-based electrostatic scoring function.


    References
    ----------
    .. [1] Gasteiger, Johann, and Mario Marsili.
       “Iterative Partial Equalization of Orbital Electronegativity—a
       Rapid Access to Atomic Charges.” Tetrahedron 36, no. 22 (January 1, 1980): 3219–28.
       https://doi.org/10.1016/0040-4020(80)80168-2.


    """

    def __init__(
        self,
        probe_mol: Mol,
        fixed_mol: Mol,
        cutoff: float = 8.0,
        k: int = 400,
    ):
        super().__init__(
            probe_mol,
            fixed_mol,
            cutoff,
            k,
        )
        self.__init_charges()

    def __init_charges(self):
        _probe_mol_charges = np.array(
            [a.GetDoubleProp("_GasteigerCharge") for a in self.probe_mol.GetAtoms()]
        )
        _fixed_mol_charges = np.array(
            [
                a.GetDoubleProp("_GasteigerCharge")
                for a in self.fixed_mol.GetAtoms()
            ]
        )
        _probe_mol_charges[np.isnan(_probe_mol_charges)] = 0.0
        _fixed_mol_charges[np.isnan(_fixed_mol_charges)] = 0.0

        self._probe_mol_charges = _probe_mol_charges[self.probe_mask]
        self._fixed_mol_charges = _fixed_mol_charges[self.fixed_mask]

    # Charge-based, not radii-based — no _kernel(dist), same reasoning as LJ above.
    _score_field = None
    _batch_scores = ScoringFunction._batch_scores


class ElectroStatic(_ChargeScoringFunction):
    r"""
    🚶 — Electrostatic force based on Gasteiger charges.

    Scores interactions based on a power of the distance and multiplication by atom charges.


    The score is calculated as:

    .. math::
        c_a \times c_b \times \min(cap, \frac{1}{r^{power}})

    Where :math:`c_a` and :math:`c_b` are the charges of the two atoms. Atoms with equal signed
    charges will thus result in positive scores, and atoms with differently signed charges will
    result in negative scores.

    These charges are calculated using the Gasteiger [1]_ method, using
    :func:`~rdkit.Chem.rdPartialCharges.ComputeGasteigerCharges`.

    This results in the following function, where in this example the `power` is 1
    and `cap` is 10.0. The blue line shows two atoms with differently signed charges, and the red
    line shows to atoms with equally signed charges.

    .. plot::
       :width: 80%
       :alt: Electrostatic example

       import numpy as np
       import matplotlib.pyplot as plt

       a = 1
       b = -1
       c = 1

       cap = 10
       power = 1

       x = np.linspace(0.0001, 6, 400)

       y1 = a * b * np.minimum(cap, 1 / (x**power))
       y2 = a * c * np.minimum(cap, 1 / (x**power))

       plt.figure()
       plt.grid(visible=True)
       plt.plot(x, y1, linewidth=2, label="$c_a = 1; c_b = -1$")
       plt.plot(x, y2, linewidth=2, color="red", label="$c_a = 1; c_b = 1$")
       plt.xlabel("Distance")
       plt.ylabel("Score")
       plt.legend(loc='upper right')


    This score is calculated for every `probe_mol` atom to all its neighbors
    (as defined by `cutoff` and `k`), and then summed.

    **Speed**:🚶–🚲

    Parameters
    ----------
    probe_mol : Mol
        The probe molecule to be used for the calculation.
    fixed_mol : Mol
        The fixed molecule to be used for the calculation.
    power : int, default 1
        A power to be applied to the distance.
    cap : float, default 100.0
        The maximum score for a single atom-atom interaction.
    cutoff : float, default 8.0
        NOT WORKING. Maximum distance to consider in the nearest neighbor search.
    k : int, default 400
        The number of neighbors to consider.
    References
    ----------
    .. [1] Gasteiger, Johann, and Mario Marsili.
       “Iterative Partial Equalization of Orbital Electronegativity—a
       Rapid Access to Atomic Charges.” Tetrahedron 36, no. 22 (January 1, 1980): 3219–28.
       https://doi.org/10.1016/0040-4020(80)80168-2.


    """

    def __init__(
        self,
        probe_mol: Mol,
        fixed_mol: Mol,
        power: int = 1,
        cap: float = 100.0,
        cutoff: float = 8.0,
        k: int = 400,
    ):
        super().__init__(
            probe_mol,
            fixed_mol,
            cutoff,
            k,
        )

        self._power = power
        self._cap = cap

    def _score(self, conf_id: int, computed: dict[Dependency, Any] | None) -> float:
        r, idx, mask = computed[self.nn_dep]
        r = r[self.probe_mask]
        idx = idx[self.probe_mask]
        mask = mask[self.probe_mask]
        safe_idx = np.where(mask, idx, 0)

        tmp = np.minimum(self._cap, 1 / (r**self._power))

        # Charge multiplier
        # ab = (self._probe_mol_charges * self._fixed_mol_charges[safe_idx].T).T
        ab = self._probe_mol_charges[:, None] * self._fixed_mol_charges[safe_idx]

        s = tmp * ab

        mask &= r < self.cutoff

        s[~mask] = 0.0

        return np.sum(s)


class AD4Solvation(_ChargeScoringFunction):
    r"""
    🚶 — Solvation force based on the AutoDesk 4 function and Gasteiger charges.

    Scores interactions based on a solvation calculation and multiplication by atom charges.


    Firstly, a distance factor :math:`d` is calculated as:

    .. math::
        d(r) = \exp(-(\frac{r}{2\sigma})^2)

    The charge-independent component is calculated as:

    .. math::
        u(r) = (solvation_1 \times volume_2 \times d(r)) + (solvation_2 \times volume_1 \times d(r))

    Charge-dependent components are described as:

    .. math::
        a(r) = |c_a| \times (q \times volume_2 \times d(r))
        b(r) = |c_b| \times (q \times volume_1 \times d(r))

    The total score is the sum of all components:

    .. math::
        u(r) + a(r) + b(r)

    Here :math:`c_a` and :math:`c_b` are the gasteiger charges [1]_ of the two atoms, calculated
    using :func:`~rdkit.Chem.rdPartialCharges.ComputeGasteigerCharges`. :math:`q` determines
    how charge-dependent the value is, and :math:`\sigma` describes the the width of the gaussian
    used.

    This results in the following function, where in this example the `solvation` of both atoms is
    -0.0005 and the `volume` of both atoms is 30. :math:`\sigma` is 3.6 and :math:`q` is 0.01097.
    The charge of the atoms is 1 and -1, respectively.

    .. plot::
       :width: 80%
       :alt: AD4Solvation example

       import numpy as np
       import matplotlib.pyplot as plt

       solv = -0.0005
       volume = 30

       sigma = 3.6
       q = 0.01097

       c_a = 1
       c_b = -1

       x = np.linspace(0.001, 6, 400)

       dist = np.exp(-np.square(x / (2 * sigma)))

       n_dep = solv * volume * dist + volume * solv * dist

       a_dep = q * volume * dist
       b_dep = q * volume * dist

       y = n_dep + (np.abs(c_a) * a_dep) + (np.abs(c_b) * b_dep)

       plt.figure()
       plt.grid(visible=True)
       plt.plot(x, y, linewidth=2)
       plt.xlabel("Distance")
       plt.ylabel("Score")


    This score is calculated for every `probe_mol` atom to all its neighbors
    (as defined by `cutoff` and `k`), and then summed.

    **Speed**: 🐢–🚲

    Parameters
    ----------
    probe_mol : Mol
        The probe molecule to be used for the calculation.
    fixed_mol : Mol
        The fixed molecule to be used for the calculation.
    d_sigma : float, default 3.6
        The width of the gaussian used.
    s_q : float, default 0.01097
        Describes how charge-dependent the score is.
    cutoff : float, default 8.0
        NOT WORKING. Maximum distance to consider in the nearest neighbor search.
    k : int, default 400
        The number of neighbors to consider.
    References
    ----------
    .. [1] Gasteiger, Johann, and Mario Marsili.
       “Iterative Partial Equalization of Orbital Electronegativity—a
       Rapid Access to Atomic Charges.” Tetrahedron 36, no. 22 (January 1, 1980): 3219–28.
       https://doi.org/10.1016/0040-4020(80)80168-2.

    """

    def __init__(
        self,
        probe_mol: Mol,
        fixed_mol: Mol,
        d_sigma: float = 3.6,
        s_q: float = 0.01097,
        cutoff: float = 8.0,
        k: int = 400,
    ):
        super().__init__(
            probe_mol,
            fixed_mol,
            cutoff,
            k,
        )

        self._d_sigma = d_sigma
        self._s_q = s_q

        self.__init_solvation()
        self.__init_volume()

    def __init_solvation(self):
        max_type = max(e.value for e in AtomType)  # noqa
        self.ad_solvation = np.empty(max_type + 1, dtype=float)
        for _, t in vina_atom_consts.items():
            self.ad_solvation[t.type.value] = t.ad_solvation

        self.probe_mol_solvation = self.ad_solvation[self.probe_mol.atom_types[self.probe_mask]]

    def __init_volume(self):
        max_type = max(e.value for e in AtomType)  # noqa
        self.ad_volume = np.empty(max_type + 1, dtype=float)
        for _, t in vina_atom_consts.items():
            self.ad_volume[t.type.value] = t.ad_volume

        self.probe_mol_volume = self.ad_volume[self.probe_mol.atom_types[self.probe_mask]]

    def _score(self, conf_id: int, computed: dict[Dependency, Any] | None) -> float:
        r, idx, mask = computed[self.nn_dep]
        r = r[self.probe_mask]
        idx = idx[self.probe_mask]
        mask = mask[self.probe_mask]
        safe_idx = np.where(mask, idx, 0)

        dist_factor = np.exp(-np.square(r / (2 * self._d_sigma)))

        fixed_mol_solvation = self.ad_solvation[
            self.fixed_mol._atom_types[self.fixed_mask][safe_idx]  # noqa
        ]
        fixed_mol_volume = self.ad_volume[
            self.fixed_mol._atom_types[self.fixed_mask][safe_idx]  # noqa
        ]

        non_charge_dep = (
            self.probe_mol_solvation[:, None] * fixed_mol_volume * dist_factor
            + self.probe_mol_volume[:, None] * fixed_mol_solvation * dist_factor
        )

        abs_a_dep = self._s_q * fixed_mol_volume * dist_factor
        abs_b_dep = self._s_q * self.probe_mol_volume[:, None] * dist_factor

        s = (
            non_charge_dep
            + (np.abs(self._probe_mol_charges)[:, None] * abs_a_dep)
            + (np.abs(self._fixed_mol_charges[safe_idx]) * abs_b_dep)
        )

        mask &= r < self.cutoff

        s[~mask] = 0.0

        return np.sum(s)


class _PLP(_KNNScoringFunction, ABC):
    r"""
    ⚙️ — Piecewise linear potential.

    Base class for piecewise linear potential calculations. Supports four-piece (van der Waals)
    and two-piece (repulsion) linear potentials.

    The potentials can be calculated using the :func:`potential_four_piece` and
    :func:`potential_two_piece` static methods.

    .. plot::
       :width: 80%
       :alt: PLP example

       import numpy as np
       import matplotlib.pyplot as plt

       x = np.linspace(3, 6, 400)

       a = 3.4
       b = 3.8
       c = 4.2
       d = 5.5
       e = -0.4
       f = 20

       y1 = np.zeros_like(x)
       y1[x < a] = (f * (a - x[x < a])) / a
       y1[(a <= x) & (x < b)] = (e * (x[(a <= x) & (x < b)] - a)) / (b - a)
       y1[(b <= x) & (x < c)] = e
       y1[(c <= x) & (x < d)] = (e * (d - x[(c <= x) & (x < d)])) / (d - c)

       a = 4
       b = 5.5
       c = 0.4
       d = 20

       y2 = np.zeros_like(x)
       y2[x < a] = x[x < a] * (c - d) / a + d
       y2[(a <= x) & (x <= b)] = -c * (x[(a <= x) & (x <= b)] - a) / (b - a) + c

       plt.figure()
       plt.grid(visible=True)
       plt.plot(x, y1, linewidth=2, label="Four piece")
       plt.plot(x, y2, linewidth=2, color="red", label="Two piece")
       plt.ylim(-1, 2)
       plt.xlabel("Distance")
       plt.ylabel("Score")
       plt.legend(loc='upper right')


    **Speed**: ⚙️, depends on implementation.

    Parameters
    ----------
    probe_mol : Mol
        The probe molecule to be used for the calculation.
    fixed_mol : Mol
        The fixed molecule to be used for the calculation.
    cutoff : float, default 8.0
        NOT WORKING. Maximum distance to consider in the nearest neighbor search.
    k : int, default 400
        The number of neighbors to consider.

    See Also
    --------
    PlantsPLP
        For a piecewise linear potential implementation like in the PLANTS docking suite.

    """

    def __init__(
        self,
        probe_mol: Mol,
        fixed_mol: Mol,
        cutoff: float = 8.0,
        k: int = 400,
    ):
        super().__init__(
            probe_mol,
            fixed_mol,
            cutoff,
            k,
        )

    # Per-pair interaction-type lookup, not a radii-relative kernel — no _kernel(dist),
    # same reasoning as LJ above.
    _score_field = None
    _batch_scores = ScoringFunction._batch_scores

    @staticmethod
    def potential_four_piece(r, values):
        r"""Calculates a four-piece linear potential based on `values`.

        The potential is calculated as follows:

        .. math::
            s =
                \begin{cases}
                    \frac{f \times (a - r)}{a}, & r < a,\\[4pt]
                    \frac{e \times (r - a)}{b - a}, & a \leq r < b,\\[4pt]
                    e, & b \leq r < c,\\[4pt]
                    \frac{e \times (d - r)}{d - c}, & c \leq r < d,\\[4pt]
                    0, & d \leq r,
                \end{cases}

        This results in the following function:

        .. plot::
           :width: 80%
           :alt: PLP example

           import numpy as np
           import matplotlib.pyplot as plt

           a = 2
           b = 3
           c = 4
           d = 6
           e = -1
           f = 5

           x = np.linspace(0, 7, 400)

           y = np.zeros_like(x)
           y[x < a] = (f * (a - x[x < a])) / a
           y[(a <= x) & (x < b)] = (e * (x[(a <= x) & (x < b)] - a)) / (b - a)
           y[(b <= x) & (x < c)] = e
           y[(c <= x) & (x < d)] = (e * (d - x[(c <= x) & (x < d)])) / (d - c)

           plt.figure()
           plt.grid(visible=True)
           plt.xticks([0, a, b, c, d], ['0', 'a', 'b', 'c', 'd'])
           plt.yticks([0, e, f], ['0', 'e', 'f'])
           plt.plot(x, y, linewidth=2)
           plt.xlabel("Distance")
           plt.ylabel("Score")

        Parameters
        ----------
        r : array_like
            An array containing distances.
        values : array_like
            The six values used for the linear potential.


        Returns
        -------
        numpy.ndarray

        """

        a, b, c, d, e, f = values
        return _four_piece_kernel(r, a, b, c, d, e, f)

    @staticmethod
    def potential_two_piece(r, values):
        r"""Calculates a two-piece linear potential based on `values`.

        The potential is calculated as follows:

        .. math::
            s =
                \begin{cases}
                    r \times \frac{c - d}{a} + d, & r < a,\\[4pt]
                    -c \times \frac{r - a}{b - a} + c, & a \leq r < b,\\[4pt]
                    0, & b \leq r,
                \end{cases}

        This results in the following function:

        .. plot::
           :width: 80%
           :alt: PLP example

           import numpy as np
           import matplotlib.pyplot as plt

           a = 2
           b = 6
           c = 1
           d = 5

           x = np.linspace(0, 7, 400)

           y = np.zeros_like(x)
           y[x < a] = x[x < a] * (c - d) / a + d
           y[(a <= x) & (x <= b)] = -c * (x[(a <= x) & (x <= b)] - a) / (b - a) + c

           plt.figure()
           plt.grid(visible=True)
           plt.plot(x, y, linewidth=2)
           plt.xticks([0, a, b], ['0', 'a', 'b'])
           plt.yticks([0, c, d], ['0', 'c', 'd'])
           plt.xlabel("Distance")
           plt.ylabel("Score")

        Parameters
        ----------
        r : array_like
            An array containing distances.
        values : array_like
            The six values used for the linear potential.


        Returns
        -------
        numpy.ndarray

        """
        a, b, c, d = values
        return _two_piece_kernel(r, a, b, c, d)


class PlantsPLP(_PLP):
    r"""
    🐢 — PLANTS piecewise linear potential.

    Piecewise linear potential as implemented in the PLANTS [1]_ docking software.

    This function defines five interaction types, where the piecewise potential parameters are
    defined as follows:

    +------------------+-----+-----+------------------------+-----------------------+-------------+----+
    | Interaction type | A   | B   | C                      | D                     | E           | F  |
    +==================+=====+=====+========================+=======================+=============+====+
    | H-bond           | 2.3 | 2.6 | 3.1                    | 3.4                   | :math:`w_0` | 20 |
    +------------------+-----+-----+------------------------+-----------------------+-------------+----+
    | Metal            | 1.4 | 2.2 | 2.6                    | 2.8                   | :math:`w_1` | 20 |
    +------------------+-----+-----+------------------------+-----------------------+-------------+----+
    | Buried           | 3.4 | 3.6 | 4.5                    | 5.5                   | :math:`w_2` | 20 |
    +------------------+-----+-----+------------------------+-----------------------+-------------+----+
    | Non-polar        | 3.4 | 3.6 | 4.5                    | 5.5                   | :math:`w_3` | 20 |
    +------------------+-----+-----+------------------------+-----------------------+-------------+----+
    | Repulsive        | 3.2 | 5.0 | :math:`w_4 \times` 0.1 | :math:`w_4 \times` 20 | -           | -  |
    +------------------+-----+-----+------------------------+-----------------------+-------------+----+

    Here, repulsive is represented by a :func:`potential_two_piece`, and the other types by a
    :func:`potential_four_piece`. :math:`w_0` to :math:`w_4` are defined as the `weights` passed into
    the function. The interaction types are assigned as described in [1]_.


    .. plot::
       :width: 80%
       :alt: PLP example

       import numpy as np
       import matplotlib.pyplot as plt

       x = np.linspace(3, 6, 400)

       a = 3.4
       b = 3.8
       c = 4.2
       d = 5.5
       e = -0.4
       f = 20

       y1 = np.zeros_like(x)
       y1[x < a] = (f * (a - x[x < a])) / a
       y1[(a <= x) & (x < b)] = (e * (x[(a <= x) & (x < b)] - a)) / (b - a)
       y1[(b <= x) & (x < c)] = e
       y1[(c <= x) & (x < d)] = (e * (d - x[(c <= x) & (x < d)])) / (d - c)

       a = 4
       b = 5.5
       c = 0.4
       d = 20

       y2 = np.zeros_like(x)
       y2[x < a] = x[x < a] * (c - d) / a + d
       y2[(a <= x) & (x <= b)] = -c * (x[(a <= x) & (x <= b)] - a) / (b - a) + c

       plt.figure()
       plt.grid(visible=True)
       plt.plot(x, y1, linewidth=2, label="PLP")
       plt.plot(x, y2, linewidth=2, color="red", label="Repulsion")
       plt.ylim(-1, 2)
       plt.xlabel("Distance")
       plt.ylabel("Score")
       plt.legend(loc='upper right')


    **Speed**: 🐜–🚶

    Parameters
    ----------
    probe_mol : Mol
        The probe molecule to be used for the calculation.
    fixed_mol : Mol
        The fixed molecule to be used for the calculation.
    weights : array_like, default (-4.0, -7.0, -0.05, -0.40, 0.50)
        A tuple or list containing the weights for each interaction type.
    cutoff : float, default 8.0
        NOT WORKING. Maximum distance to consider in the nearest neighbor search.
    k : int, default 400
        The number of neighbors to consider.

    References
    ----------
    .. [1] Korb, Oliver, Thomas Stützle, and Thomas E. Exner.
        “Empirical Scoring Functions for Advanced Protein−Ligand Docking with PLANTS.”
        Journal of Chemical Information and Modeling 49, no. 1 (January 26, 2009): 84–96.
        https://doi.org/10.1021/ci800298z.

    """

    def __init__(
        self,
        probe_mol: Mol,
        fixed_mol: Mol,
        weights: tuple[float, float, float, float, float] | None = None,
        cutoff: float = 8.0,
        k: int = 400,
    ):
        super().__init__(
            probe_mol,
            fixed_mol,
            cutoff,
            k,
        )
        if weights is None:
            weights = (-4.0, -7.0, -0.05, -0.40, 0.50)
        self.weights = weights
        w0, w1, w2, w3, w4 = weights

        self.__init_plants_interaction_types()
        self.parameters = {
            PlantsPLP._InteractionType.HBOND: (2.3, 2.6, 3.1, 3.4, w0, 20.0),
            PlantsPLP._InteractionType.METAL: (1.4, 2.2, 2.6, 2.8, w1, 20.0),
            PlantsPLP._InteractionType.BURIED: (3.4, 3.6, 4.5, 5.5, w2, 20.0),
            PlantsPLP._InteractionType.NONPOLAR: (3.4, 3.6, 4.5, 5.5, w3, 20.0),
            PlantsPLP._InteractionType.REPULSIVE: (3.2, 5.0, w4 * 0.1, w4 * 20.0),
        }

    class _InteractionType(IntEnum):
        UNKNOWN = 0
        REPULSIVE = 1
        HBOND = 2
        BURIED = 3
        NONPOLAR = 4
        METAL = 5

    def __init_plants_interaction_types(self):
        # construct lut for (probe_mol_atom, fixed_mol_atom)

        is_donor = lambda x: (  # noqa
            (x == AtomType.NitrogenDonor) | (x == AtomType.OxygenDonor)
        )
        is_acceptor = lambda x: (  # noqa
            (x == AtomType.NitrogenAcceptor) | (x == AtomType.OxygenAcceptor)
        )
        is_donacc = lambda x: (  # noqa
            (x == AtomType.NitrogenDonorAcceptor) | (x == AtomType.OxygenDonorAcceptor)
        )
        is_metal = lambda x: (  # noqa
            (x == AtomType.GenericMetal)
            | (x == AtomType.Magnesium)
            | (x == AtomType.Manganese)
            | (x == AtomType.Zinc)
            | (x == AtomType.Calcium)
            | (x == AtomType.Iron)
        )
        is_nonpolar = lambda x: (  # noqa
            (x != AtomType.Hydrogen)
            & (x != AtomType.PolarHydrogen)
            & (~is_donor(x))
            & (~is_acceptor(x))
            & (~is_donacc(x))
            & (~is_metal(x))
        )

        lig_types = self.probe_mol.atom_types[self.probe_mask]
        rec_types = self.fixed_mol.atom_types[self.fixed_mask]

        lut = np.zeros(
            (lig_types.shape[0], rec_types.shape[0]),
            dtype=np.int32,
        )

        self.mask_rep = (
            (is_donor(lig_types)[:, None] & is_donor(rec_types)[None, :])
            | (is_acceptor(lig_types)[:, None] & is_acceptor(rec_types)[None, :])
            | (is_donor(lig_types)[:, None] & is_metal(rec_types)[None, :])
        )

        self.mask_hb = (
            (
                is_donor(lig_types)[:, None]
                & ((is_acceptor(rec_types) | is_donacc(rec_types))[None, :])
            )
            | (
                is_acceptor(lig_types)[:, None]
                & (is_donor(rec_types) | is_donacc(rec_types))[None, :]
            )
            | (
                is_donacc(lig_types)[:, None]
                & (is_donor(rec_types) | is_acceptor(rec_types) | is_donacc(rec_types))[
                    None, :
                ]
            )
        )

        self.mask_buried = (
            (is_donor(lig_types) | is_acceptor(lig_types) | is_donacc(lig_types))[
                :, None
            ]
            & is_nonpolar(rec_types)[None, :]
        ) | (
            is_nonpolar(lig_types)[:, None]
            & (
                is_donor(rec_types)
                | is_acceptor(rec_types)
                | is_donacc(rec_types)
                | is_metal(rec_types)
            )[None, :]
        )

        self.mask_np = is_nonpolar(lig_types)[:, None] & is_nonpolar(rec_types)[None, :]

        self.mask_metal = (is_acceptor(lig_types) | is_donacc(lig_types))[
            :, None
        ] & is_metal(rec_types)[None, :]

        lut[self.mask_rep] = PlantsPLP._InteractionType.REPULSIVE
        lut[self.mask_hb] = PlantsPLP._InteractionType.HBOND
        lut[self.mask_buried] = PlantsPLP._InteractionType.BURIED
        lut[self.mask_np] = PlantsPLP._InteractionType.NONPOLAR
        lut[self.mask_metal] = PlantsPLP._InteractionType.METAL

        self.interaction_types = lut

    def _score(self, conf_id: int, computed: dict[Dependency, Any] | None) -> float:
        r, idx, mask = computed[self.nn_dep]
        r = r[self.probe_mask]
        idx = idx[self.probe_mask]
        mask = mask[self.probe_mask]
        safe_idx = np.where(mask, idx, 0)

        interact_types = np.take_along_axis(
            self.interaction_types, safe_idx, axis=1
        )  # M <3 J

        s = np.zeros_like(r)

        for i_type in {
            PlantsPLP._InteractionType.HBOND,
            PlantsPLP._InteractionType.BURIED,
            PlantsPLP._InteractionType.NONPOLAR,
            PlantsPLP._InteractionType.METAL,
        }:
            s[interact_types == i_type] = self.potential_four_piece(
                r[interact_types == i_type], self.parameters[i_type]
            )
        s[interact_types == PlantsPLP._InteractionType.REPULSIVE] = (
            self.potential_two_piece(
                r[interact_types == PlantsPLP._InteractionType.REPULSIVE],
                self.parameters[PlantsPLP._InteractionType.REPULSIVE],
            )
        )

        return np.sum(s)
