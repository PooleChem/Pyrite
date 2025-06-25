from abc import ABC
from enum import IntEnum
from typing import Any

import numpy as np

from .._common import Ligand, Receptor
from ..atom_consts import AtomType, vina_atom_consts
from ._base import ScoringFunction
from .dependencies import Dependency, KNNDependency


class _KNNScoringFunction(ScoringFunction, ABC):
    r"""
    ⚙️ — K-Nearest neighbor distance scoring function.

    This abstract scoring function can be used to implement scoring functions that make use of the
    distance of ligand atoms to the closest `k` protein atoms.

    .. note::
        This is an abstract base class and should thus be subclassed. Please refer to TODO
         for more information on how to do this.


    **Speed**: ⚙️, depends on implementation.

    Parameters
    ----------
    ligand : Ligand
        The ligand to be used for the calculation.
    receptor : Receptor
        The receptor to be used for the calculation.
    cutoff : float, default 8.0
        NOT WORKING. Maximum distance to consider in the nearest neighbor search.
    k : int, default 400
        The number of neighbors to consider.
    ignore_non_polar_hydrogens : bool, default True
        If `ignore_non_polar_hydrogens` is ``True``, non-polar hydrogen atoms are masked out. The
        masking happens at initialization, so `k` is not impacted.


    See Also
    --------
    _ChargeScoringFunction
        Abstract charge-dependent scoring function.

    """

    def __init__(
        self,
        ligand: Ligand,
        receptor: Receptor,
        cutoff: float = 8.0,
        k: int = 400,
        ignore_non_polar_hydrogens: bool = True,
    ):
        self.ligand = ligand
        self.receptor = receptor

        self.lig_mask = ~(
            ignore_non_polar_hydrogens
            & (
                (self.ligand.atom_types == AtomType.Hydrogen)
                | (self.ligand.atom_types == AtomType.PolarHydrogen)
            )
        )

        self.rec_mask = ~(
            ignore_non_polar_hydrogens
            & (
                (self.receptor._atom_types == AtomType.Hydrogen)
                | (self.receptor._atom_types == AtomType.PolarHydrogen)
            )
        )

        self.cutoff = cutoff
        self.k = k

        self.__init_radii()

        self.nn_dep = KNNDependency(
            hash((receptor, ignore_non_polar_hydrogens)),
            self.receptor.positions[self.rec_mask],
            self.ligand.get_positions,
            self.k,
            self.cutoff,
        )
        self._tree_n = len(self.receptor.positions)

    def __init_radii(self):
        max_type = max(e.value for e in AtomType)
        self.xs_radii = np.empty(max_type + 1, dtype=float)
        for _, t in vina_atom_consts.items():
            self.xs_radii[t.type.value] = t.xs_radius

        self.ligand_radii = self.xs_radii[self.ligand.atom_types[self.lig_mask]]

    def get_dependencies(self) -> set[Dependency]:
        return {self.nn_dep}

    def _get_optimal_distance(self, idx, mask):
        safe_idx = np.where(mask, idx, 0)
        # print(
        #     mask.shape,
        #     idx.shape,
        #     safe_idx.shape,
        #     self.rec_mask.shape,
        #     np.sum(self.rec_mask),
        #     self.receptor.atom_types.shape,
        #     self.receptor.atom_types[self.rec_mask].shape,
        #     np.max(safe_idx),
        #     self.xs_radii.shape,
        # )
        receptor_radii = self.xs_radii[
            self.receptor.atom_types[self.rec_mask][safe_idx]
        ]

        return self.ligand_radii[:, None] + receptor_radii[:, :]


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

    This score is calculated for every ligand atom to all its neighbors
    (as defined by `cutoff` and `k`), and then summed.

    **Speed**: 🐢–🚗, depending on `cutoff` and `k`.

    Parameters
    ----------
    ligand : Ligand
        The ligand to be used for the calculation.
    receptor : Receptor
        The receptor to be used for the calculation.
    offset : float, default 0.0
        The offset from `optimal_distance` that is considered as ideal.
    width : float, default 0.5
        The width of the Gaussian.
    cutoff : float, default 8.0
        NOT WORKING. Maximum distance to consider in the nearest neighbor search.
    k : int, default 400
        The number of neighbors to consider. It is necessary to increase this number if a wider or
        higher offset Gaussian is used.
    ignore_non_polar_hydrogens : bool, default True
        If `ignore_non_polar_hydrogens` is ``True``, non-polar hydrogen atoms are masked out. The
        masking happens at initialization, so `k` is not impacted.


    """

    def __init__(
        self,
        ligand: Ligand,
        receptor: Receptor,
        offset: float = 0.0,
        width: float = 0.5,
        cutoff: float = None,
        k: int = 400,
        ignore_non_polar_hydrogens: bool = True,
    ):
        super().__init__(
            ligand,
            receptor,
            cutoff or 4 + offset + np.e * width,
            k,
            ignore_non_polar_hydrogens,
        )
        self.offset = offset
        self.width = width

    @staticmethod
    def __gaussian(x, w):
        return np.exp(-((x / w) ** 2))

    def _score(self, conf_id, computed) -> float:
        r, idx, mask = computed[self.nn_dep]
        r = r[self.lig_mask]
        idx = idx[self.lig_mask]
        mask = mask[self.lig_mask]

        optimal = self._get_optimal_distance(idx, mask) + self.offset
        s = self.__gaussian(r - optimal, self.width)

        mask &= r < self.cutoff

        s[~mask] = 0.0

        return np.sum(s)


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

    This score is calculated for every ligand atom to all its neighbors
    (as defined by `cutoff` and `k`), and then summed.

    **Speed**:🚲

    Parameters
    ----------
    ligand : Ligand
        The ligand to be used for the calculation.
    receptor : Receptor
        The receptor to be used for the calculation.
    offset : float, default 0.0
        The offset that is added to `optimal_distance`.
    cutoff : float, default 8.0
        NOT WORKING. Maximum distance to consider in the nearest neighbor search.
    k : int, default 400
        The number of neighbors to consider.
    ignore_non_polar_hydrogens : bool, default True
        If `ignore_non_polar_hydrogens` is ``True``, non-polar hydrogen atoms are masked out. The
        masking happens at initialization, so `k` is not impacted.

    """

    def __init__(
        self,
        ligand: Ligand,
        receptor: Receptor,
        offset: float = 0.0,
        cutoff: float = None,
        k: int = 400,
        ignore_non_polar_hydrogens: bool = True,
    ):
        super().__init__(
            ligand, receptor, cutoff or 4 + offset, k, ignore_non_polar_hydrogens
        )
        self.offset = offset

    def _score(self, conf_id, computed) -> float:
        r, idx, mask = computed[self.nn_dep]
        r = r[self.lig_mask]
        idx = idx[self.lig_mask]
        mask = mask[self.lig_mask]

        optimal = self._get_optimal_distance(idx, mask) + self.offset
        d = r - optimal

        d[~mask | (d > 0.0)] = 0.0

        return np.sum(d * d)


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
    ligand : Ligand
        The ligand to be used for the calculation.
    receptor : Receptor
        The receptor to be used for the calculation.
    good : float, default 0.5
        The `good` distance. Distance values lower than this value are scored :math:`1`.
    bad : float, default 1.5
        The `bad` distance. Distance values higher than this value are scored :math:`0`.
    cutoff : float, default 8.0
        NOT WORKING. Maximum distance to consider in the nearest neighbor search.
    k : int, default 400
        The number of neighbors to consider.
    ignore_non_polar_hydrogens : bool, default True
        If `ignore_non_polar_hydrogens` is ``True``, non-polar hydrogen atoms are masked out. The
        masking happens at initialization, so `k` is not impacted.


    See Also
    --------
    _ChargeScoringFunction
        Abstract charge-dependent scoring function.

    """

    def __init__(
        self,
        ligand: Ligand,
        receptor: Receptor,
        good: float = 0.5,
        bad: float = 1.5,
        cutoff: float = None,
        k: int = 400,
        ignore_non_polar_hydrogens: bool = True,
    ):
        assert good < bad, "Bad distance <= good distance not implemented."
        super().__init__(ligand, receptor, cutoff or bad, k, ignore_non_polar_hydrogens)
        self.good = good
        self.bad = bad

    @staticmethod
    def _slope_step(dist, good, bad):
        slope_step = (dist - bad) / (good - bad)
        slope_step[dist >= bad] = 0.0
        slope_step[dist <= good] = 1.0
        return slope_step

    def _mask(self, receptor_val, neighbor_mask):
        return np.full(receptor_val.shape, True) & neighbor_mask

    def _score(self, conf_id, computed) -> float:
        r, idx, neighbor_mask = computed[self.nn_dep]
        r = r[self.lig_mask]
        idx = idx[self.lig_mask]
        neighbor_mask = neighbor_mask[self.lig_mask]

        optimal_distance = self._get_optimal_distance(idx, neighbor_mask)
        dist = r - optimal_distance

        slope_step = self._slope_step(dist, self.good, self.bad)

        mask = self._mask(idx, neighbor_mask)
        slope_step[~mask] = 0.0

        return np.sum(slope_step)


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

    This score is calculated for every hydrophobic ligand atom to all its hydrophobic neighbors
    (as defined by `cutoff` and `k`), and then summed. The hydrophobic neighbors are determined
    after the nearest neighbor search.

    **Speed**:🚲

    Parameters
    ----------
    ligand : Ligand
        The ligand to be used for the calculation.
    receptor : Receptor
        The receptor to be used for the calculation.
    good : float, default 0.5
        The start of the `slope-step`.
    bad : float, default 1.5
        The end of the `slope-step`.
    cutoff : float, default 8.0
        NOT WORKING. Maximum distance to consider in the nearest neighbor search.
    k : int, default 400
        The number of neighbors to consider.
    ignore_non_polar_hydrogens : bool, default True
        If `ignore_non_polar_hydrogens` is ``True``, non-polar hydrogen atoms are masked out. The
        masking happens at initialization, so `k` is not impacted.


    See Also
    --------
    NonHydrophobic
        For hydrophilic interactions.

    """

    def __init__(
        self,
        ligand: Ligand,
        receptor: Receptor,
        good: float = 0.5,
        bad: float = 1.5,
        cutoff: float = None,
        k: int = 400,
        ignore_non_polar_hydrogens: bool = True,
    ):
        super().__init__(
            ligand, receptor, good, bad, cutoff or bad, k, ignore_non_polar_hydrogens
        )

        self.__init_hydrophobic()

    def __init_hydrophobic(self):
        max_type = max(e.value for e in AtomType)
        self.xs_hydrophobic = np.empty(max_type + 1, dtype=bool)
        for _, t in vina_atom_consts.items():
            self.xs_hydrophobic[t.type.value] = t.xs_hydrophobe

        self.ligand_hydrophobic = self.xs_hydrophobic[
            self.ligand.atom_types[self.lig_mask]
        ]

    def _mask(self, idx, neighbor_mask):
        safe_idx = np.where(neighbor_mask, idx, 0)
        receptor_hydrophobic = self.xs_hydrophobic[
            self.receptor._atom_types[self.rec_mask][safe_idx]
        ]
        return (
            self.ligand_hydrophobic[:, None]
            & receptor_hydrophobic[:, :]
            & neighbor_mask
        )


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

    This score is calculated for every nonhydrophobic ligand atom to all its
    nonhydrophobic neighbors (as defined by `cutoff` and `k`), and then summed.
    The nonhydrophobic neighbors are determined after the nearest neighbor search.

    **Speed**:🚲

    Parameters
    ----------
    ligand : Ligand
        The ligand to be used for the calculation.
    receptor : Receptor
        The receptor to be used for the calculation.
    good : float, default 0.5
        The start of the `slope-step`.
    bad : float, default 1.5
        The end of the `slope-step`.
    cutoff : float, default 8.0
        NOT WORKING. Maximum distance to consider in the nearest neighbor search.
    k : int, default 400
        The number of neighbors to consider.
    ignore_non_polar_hydrogens : bool, default True
        If `ignore_non_polar_hydrogens` is ``True``, non-polar hydrogen atoms are masked out. The
        masking happens at initialization, so `k` is not impacted.


    See Also
    --------
    Hydrophobic
        For hydrophobic interactions.

    """

    def _mask(self, idx, neighbor_mask):
        safe_idx = np.where(neighbor_mask, idx, 0)
        receptor_hydrophobic = self.xs_hydrophobic[
            self.receptor._atom_types[self.rec_mask][safe_idx]
        ]
        return ~self.ligand_hydrophobic[:, None] & ~receptor_hydrophobic & neighbor_mask


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

    This score is calculated for every hbond-acceptor ligand atom to all its hbond-donor neighbors
    and from every hbond-donor ligand atom to all its hbond-acceptor neighbors
    (as defined by `cutoff` and `k`), and then summed. The hbond neighbors are determined
    after the nearest neighbor search.

    **Speed**:🚲

    Parameters
    ----------
    ligand : Ligand
        The ligand to be used for the calculation.
    receptor : Receptor
        The receptor to be used for the calculation.
    good : float, default -0.7
        The start of the `slope-step`.
    bad : float, default 0
        The end of the `slope-step`.
    cutoff : float, default 8.0
        NOT WORKING. Maximum distance to consider in the nearest neighbor search.
    k : int, default 400
        The number of neighbors to consider.
    ignore_non_polar_hydrogens : bool, default True
        If `ignore_non_polar_hydrogens` is ``True``, non-polar hydrogen atoms are masked out. The
        masking happens at initialization, so `k` is not impacted.

    See Also
    --------
    NonDirHBondLJ
        For a hydrogen bond implementation using a Lennard-Jones potential.

    """

    def __init__(
        self,
        ligand: Ligand,
        receptor: Receptor,
        good: float = -0.7,
        bad: float = 0,
        cutoff: float = None,
        k: int = 400,
        ignore_non_polar_hydrogens: bool = True,
    ):
        super().__init__(
            ligand, receptor, good, bad, cutoff, k, ignore_non_polar_hydrogens
        )

        self.__init_hbond_possible()

    def __init_hbond_possible(self):
        max_type = max(e.value for e in AtomType)
        self.xs_acceptor = np.empty(max_type + 1, dtype=bool)
        self.xs_donor = np.empty(max_type + 1, dtype=bool)
        for _, t in vina_atom_consts.items():
            self.xs_acceptor[t.type.value] = t.xs_acceptor
            self.xs_donor[t.type.value] = t.xs_donor

        self.ligand_acceptor = self.xs_acceptor[self.ligand.atom_types[self.lig_mask]]
        self.ligand_donor = self.xs_donor[self.ligand.atom_types[self.lig_mask]]

    def _mask(self, idx, neighbor_mask):
        safe_idx = np.where(neighbor_mask, idx, 0)

        receptor_acceptor = self.xs_acceptor[
            self.receptor._atom_types[self.rec_mask][safe_idx]
        ]
        receptor_donor = self.xs_donor[
            self.receptor._atom_types[self.rec_mask][safe_idx]
        ]
        return (self.ligand_donor[:, None] & receptor_acceptor) | (
            self.ligand_acceptor[:, None] & receptor_donor
        ) & neighbor_mask


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

    This score is calculated for every ligand atom to all its neighbors
    (as defined by `cutoff` and `k`), and then summed.

    **Speed**: 🐢–🚶, depending on `cutoff` and `k`.

    Parameters
    ----------
    ligand : Ligand
        The ligand to be used for the calculation.
    receptor : Receptor
        The receptor to be used for the calculation.
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
    ignore_non_polar_hydrogens : bool, default True
        If `ignore_non_polar_hydrogens` is ``True``, non-polar hydrogen atoms are masked out. The
        masking happens at initialization, so `k` is not impacted.


    """

    def __init__(
        self,
        ligand: Ligand,
        receptor: Receptor,
        i: int = 10,
        j: int = 12,
        smoothing: float = 0,
        offset: float = 0.0,
        cap: float = 100.0,
        depth: float = 1.0,
        cutoff: float = None,
        k: int = 400,
        ignore_non_polar_hydrogens: bool = True,
    ):
        super().__init__(
            ligand,
            receptor,
            cutoff or 8 + offset,
            k,
            ignore_non_polar_hydrogens,
        )

        self._i = i
        self._j = j
        self._smoothing = smoothing
        self._offset = offset
        self._cap = cap
        self._depth = depth

    def _mask(self, receptor_val, neighbor_mask):
        return np.full(receptor_val.shape, True) & neighbor_mask

    def _score(self, conf_id, computed) -> float:
        r, idx, neighbor_mask = computed[self.nn_dep]
        r = r[self.lig_mask]
        idx = idx[self.lig_mask]
        neighbor_mask = neighbor_mask[self.lig_mask]

        optimal_distance = self._get_optimal_distance(idx, neighbor_mask) + self._offset

        c_i = (optimal_distance**self._i) * self._depth * self._j / (self._i - self._j)
        c_j = (optimal_distance**self._j) * self._depth * self._i / (self._j - self._i)

        r2 = np.full(r.shape, optimal_distance)
        mask_upper = r > (optimal_distance + self._smoothing)
        mask_lower = r < (optimal_distance - self._smoothing)
        r2[mask_upper] = r[mask_upper] - self._smoothing
        r2[mask_lower] = r[mask_lower] + self._smoothing

        r_i = r2**self._i
        r_j = r2**self._j

        res = np.minimum(self._cap, c_i / r_i + c_j / r_j)

        mask = self._mask(idx, neighbor_mask)
        res[~mask] = 0.0

        return np.sum(res)


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

    This score is calculated for every ligand atom to all its neighbors
    (as defined by `cutoff` and `k`), and then summed.

    **Speed**: 🐢–🚶

    Parameters
    ----------
    ligand : Ligand
        The ligand to be used for the calculation.
    receptor : Receptor
        The receptor to be used for the calculation.
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
    ignore_non_polar_hydrogens : bool, default True
        If `ignore_non_polar_hydrogens` is ``True``, non-polar hydrogen atoms are masked out. The
        masking happens at initialization, so `k` is not impacted.

    See Also
    --------
    LJ
        For the Lennard-Jones potential.

    """

    def __init__(
        self,
        ligand: Ligand,
        receptor: Receptor,
        i: int = 4,
        j: int = 8,
        smoothing: float = 0,
        cap: float = 100.0,
        cutoff: float = None,
        k: int = 400,
        ignore_non_polar_hydrogens: bool = True,
    ):
        super().__init__(
            ligand,
            receptor,
            i=i,
            j=j,
            smoothing=smoothing,
            offset=0,
            cap=cap,
            cutoff=cutoff,
            k=k,
            ignore_non_polar_hydrogens=ignore_non_polar_hydrogens,
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

    This score is calculated for every hbond-acceptor ligand atom to all its hbond-donor neighbors
    and from every hbond-donor ligand atom to all its hbond-acceptor neighbors
    (as defined by `cutoff` and `k`), and then summed. The hbond neighbors are determined
    after the nearest neighbor search.

    **Speed**: 🐢–🚶

    Parameters
    ----------
    ligand : Ligand
        The ligand to be used for the calculation.
    receptor : Receptor
        The receptor to be used for the calculation.
    offset : float, default -0.7
        The offset from the optimal distance.
    cap : float, default 100.0
        The maximum score for a single atom-atom interaction.
    cutoff : float, default 8.0
        NOT WORKING. Maximum distance to consider in the nearest neighbor search.
    k : int, default 400
        The number of neighbors to consider.
    ignore_non_polar_hydrogens : bool, default True
        If `ignore_non_polar_hydrogens` is ``True``, non-polar hydrogen atoms are masked out. The
        masking happens at initialization, so `k` is not impacted.

    See Also
    --------
    NonDirHBond
        For a hydrogen bond implementation using a slope-step.

    """

    def __init__(
        self,
        ligand: Ligand,
        receptor: Receptor,
        offset: float = -0.7,
        cap: float = 100.0,
        cutoff: float = None,
        k: int = 400,
        ignore_non_polar_hydrogens: bool = True,
    ):
        super().__init__(
            ligand,
            receptor,
            i=10,
            j=12,
            smoothing=0,
            offset=offset,
            cap=cap,
            depth=5.0,
            cutoff=cutoff,
            k=k,
            ignore_non_polar_hydrogens=ignore_non_polar_hydrogens,
        )
        self.__init_hbond_possible()

    def __init_hbond_possible(self):
        max_type = max(e.value for e in AtomType)
        self.xs_acceptor = np.empty(max_type + 1, dtype=bool)
        self.xs_donor = np.empty(max_type + 1, dtype=bool)
        for _, t in vina_atom_consts.items():
            self.xs_acceptor[t.type.value] = t.xs_acceptor
            self.xs_donor[t.type.value] = t.xs_donor

        self.ligand_acceptor = self.xs_acceptor[self.ligand.atom_types[self.lig_mask]]
        self.ligand_donor = self.xs_donor[self.ligand.atom_types[self.lig_mask]]

    def _mask(self, idx, neighbor_mask):
        safe_idx = np.where(neighbor_mask, idx, 0)

        receptor_acceptor = self.xs_acceptor[
            self.receptor._atom_types[self.rec_mask][safe_idx]
        ]
        receptor_donor = self.xs_donor[
            self.receptor._atom_types[self.rec_mask][safe_idx]
        ]
        return (self.ligand_donor[:, None] & receptor_acceptor) | (
            self.ligand_acceptor[:, None] & receptor_donor
        ) & neighbor_mask


class _ChargeScoringFunction(_KNNScoringFunction, ABC):
    r"""
    ⚙️ — Gasteiger-charge based scoring

    This abstract scoring function can be used to implement scoring functions that make use of the
    distance of ligand atoms to the closest `k` protein atoms, and the charge of these atoms.

    These charges are calculated using the Gasteiger [1]_ method, using
    :func:`~rdkit.Chem.rdPartialCharges.ComputeGasteigerCharges`.

    .. note::
        This is an abstract base class and should thus be subclassed. Please refer to TODO
         for more information on how to do this.


    **Speed**: ⚙️, depends on implementation.

    Parameters
    ----------
    ligand : Ligand
        The ligand to be used for the calculation.
    receptor : Receptor
        The receptor to be used for the calculation.
    cutoff : float, default 8.0
        NOT WORKING. Maximum distance to consider in the nearest neighbor search.
    k : int, default 400
        The number of neighbors to consider.
    ignore_non_polar_hydrogens : bool, default True
        If `ignore_non_polar_hydrogens` is ``True``, non-polar hydrogen atoms are masked out. The
        masking happens at initialization, so `k` is not impacted.


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
        ligand: Ligand,
        receptor: Receptor,
        cutoff: float = 8.0,
        k: int = 400,
        ignore_non_polar_hydrogens: bool = True,
    ):
        super().__init__(
            ligand,
            receptor,
            cutoff,
            k,
            ignore_non_polar_hydrogens,
        )
        self.__init_charges()

    def __init_charges(self):
        _ligand_charges = np.array(
            [a.GetDoubleProp("_GasteigerCharge") for a in self.ligand.GetAtoms()]
        )
        _receptor_charges = np.array(
            [
                a.GetDoubleProp("_GasteigerCharge")
                for a in self.receptor._rdkit.GetAtoms()
            ]
        )
        _ligand_charges[np.isnan(_ligand_charges)] = 0.0
        _receptor_charges[np.isnan(_receptor_charges)] = 0.0

        self._ligand_charges = _ligand_charges[self.lig_mask]
        self._receptor_charges = _receptor_charges[self.rec_mask]


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


    This score is calculated for every ligand atom to all its neighbors
    (as defined by `cutoff` and `k`), and then summed.

    **Speed**:🚶–🚲

    Parameters
    ----------
    ligand : Ligand
        The ligand to be used for the calculation.
    receptor : Receptor
        The receptor to be used for the calculation.
    power : int, default 1
        A power to be applied to the distance.
    cap : float, default 100.0
        The maximum score for a single atom-atom interaction.
    cutoff : float, default 8.0
        NOT WORKING. Maximum distance to consider in the nearest neighbor search.
    k : int, default 400
        The number of neighbors to consider.
    ignore_non_polar_hydrogens : bool, default True
        If `ignore_non_polar_hydrogens` is ``True``, non-polar hydrogen atoms are masked out. The
        masking happens at initialization, so `k` is not impacted.

    References
    ----------
    .. [1] Gasteiger, Johann, and Mario Marsili.
       “Iterative Partial Equalization of Orbital Electronegativity—a
       Rapid Access to Atomic Charges.” Tetrahedron 36, no. 22 (January 1, 1980): 3219–28.
       https://doi.org/10.1016/0040-4020(80)80168-2.


    """

    def __init__(
        self,
        ligand: Ligand,
        receptor: Receptor,
        power: int = 1,
        cap: float = 100.0,
        cutoff: float = 8.0,
        k: int = 400,
        ignore_non_polar_hydrogens: bool = True,
    ):
        super().__init__(
            ligand,
            receptor,
            cutoff,
            k,
            ignore_non_polar_hydrogens,
        )

        self._power = power
        self._cap = cap

    def _score(self, conf_id: int, computed: dict[Dependency, Any] | None) -> float:
        r, idx, mask = computed[self.nn_dep]
        r = r[self.lig_mask]
        idx = idx[self.lig_mask]
        mask = mask[self.lig_mask]
        safe_idx = np.where(mask, idx, 0)

        tmp = np.minimum(self._cap, 1 / (r**self._power))

        # Charge multiplier
        # ab = (self._ligand_charges * self._receptor_charges[safe_idx].T).T
        ab = self._ligand_charges[:, None] * self._receptor_charges[safe_idx]

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


    This score is calculated for every ligand atom to all its neighbors
    (as defined by `cutoff` and `k`), and then summed.

    **Speed**: 🐢–🚲

    Parameters
    ----------
    ligand : Ligand
        The ligand to be used for the calculation.
    receptor : Receptor
        The receptor to be used for the calculation.
    d_sigma : float, default 3.6
        The width of the gaussian used.
    s_q : float, default 0.01097
        Describes how charge-dependent the score is.
    cutoff : float, default 8.0
        NOT WORKING. Maximum distance to consider in the nearest neighbor search.
    k : int, default 400
        The number of neighbors to consider.
    ignore_non_polar_hydrogens : bool, default True
        If `ignore_non_polar_hydrogens` is ``True``, non-polar hydrogen atoms are masked out. The
        masking happens at initialization, so `k` is not impacted.

    References
    ----------
    .. [1] Gasteiger, Johann, and Mario Marsili.
       “Iterative Partial Equalization of Orbital Electronegativity—a
       Rapid Access to Atomic Charges.” Tetrahedron 36, no. 22 (January 1, 1980): 3219–28.
       https://doi.org/10.1016/0040-4020(80)80168-2.

    """

    def __init__(
        self,
        ligand: Ligand,
        receptor: Receptor,
        d_sigma: int = 3.6,
        s_q: float = 0.01097,
        cutoff: float = 8.0,
        k: int = 400,
        ignore_non_polar_hydrogens: bool = True,
    ):
        super().__init__(
            ligand,
            receptor,
            cutoff,
            k,
            ignore_non_polar_hydrogens,
        )

        self._d_sigma = d_sigma
        self._s_q = s_q

        self.__init_solvation()
        self.__init_volume()

    def __init_solvation(self):
        max_type = max(e.value for e in AtomType)  # noqa
        self.ad_solvation = np.empty(max_type + 1, dtype=bool)
        for _, t in vina_atom_consts.items():
            self.ad_solvation[t.type.value] = t.ad_solvation

        self.ligand_solvation = self.ad_solvation[self.ligand.atom_types[self.lig_mask]]

    def __init_volume(self):
        max_type = max(e.value for e in AtomType)  # noqa
        self.ad_volume = np.empty(max_type + 1, dtype=bool)
        for _, t in vina_atom_consts.items():
            self.ad_volume[t.type.value] = t.ad_volume

        self.ligand_volume = self.ad_volume[self.ligand.atom_types[self.lig_mask]]

    def _score(self, conf_id: int, computed: dict[Dependency, Any] | None) -> float:
        r, idx, mask = computed[self.nn_dep]
        r = r[self.lig_mask]
        idx = idx[self.lig_mask]
        mask = mask[self.lig_mask]
        safe_idx = np.where(mask, idx, 0)

        dist_factor = np.exp(-np.square(r / (2 * self._d_sigma)))

        receptor_solvation = self.ad_solvation[
            self.receptor._atom_types[self.rec_mask][safe_idx]  # noqa
        ]
        receptor_volume = self.ad_volume[
            self.receptor._atom_types[self.rec_mask][safe_idx]  # noqa
        ]

        non_charge_dep = (
            self.ligand_solvation[:, None] * receptor_volume * dist_factor
            + self.ligand_volume[:, None] * receptor_solvation * dist_factor
        )

        abs_a_dep = self._s_q * receptor_volume * dist_factor
        abs_b_dep = self._s_q * self.ligand_volume[:, None] * dist_factor

        s = (
            non_charge_dep
            + (np.abs(self._ligand_charges)[:, None] * abs_a_dep)
            + (np.abs(self._receptor_charges[safe_idx]) * abs_b_dep)
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
    ligand : Ligand
        The ligand to be used for the calculation.
    receptor : Receptor
        The receptor to be used for the calculation.
    cutoff : float, default 8.0
        NOT WORKING. Maximum distance to consider in the nearest neighbor search.
    k : int, default 400
        The number of neighbors to consider.
    ignore_non_polar_hydrogens : bool, default True
        If `ignore_non_polar_hydrogens` is ``True``, non-polar hydrogen atoms are masked out. The
        masking happens at initialization, so `k` is not impacted.


    See Also
    --------
    PlantsPLP
        For a piecewise linear potential implementation like in the PLANTS docking suite.

    """

    def __init__(
        self,
        ligand: Ligand,
        receptor: Receptor,
        cutoff: float = 8.0,
        k: int = 400,
        ignore_non_polar_hydrogens: bool = True,
    ):
        super().__init__(
            ligand,
            receptor,
            cutoff,
            k,
            ignore_non_polar_hydrogens,
        )

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

        res = np.zeros_like(r)

        res[r < a] = (f * (a - r[r < a])) / a
        res[(a <= r) & (r < b)] = (e * (r[(a <= r) & (r < b)] - a)) / (b - a)
        res[(b <= r) & (r < c)] = e
        res[(c <= r) & (r < d)] = (e * (d - r[(c <= r) & (r < d)])) / (d - c)

        return res

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

        res = np.zeros_like(r)
        res[r < a] = r[r < a] * (c - d) / a + d
        res[(a <= r) & (r <= b)] = -c * (r[(a <= r) & (r <= b)] - a) / (b - a) + c

        return res


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
    ligand : Ligand
        The ligand to be used for the calculation.
    receptor : Receptor
        The receptor to be used for the calculation.
    weights : array_like, default (-4.0, -7.0, -0.05, -0.40, 0.50)
        A tuple or list containing the weights for each interaction type.
    cutoff : float, default 8.0
        NOT WORKING. Maximum distance to consider in the nearest neighbor search.
    k : int, default 400
        The number of neighbors to consider.
    ignore_non_polar_hydrogens : bool, default True
        If `ignore_non_polar_hydrogens` is ``True``, non-polar hydrogen atoms are masked out. The
        masking happens at initialization, so `k` is not impacted.


    References
    ----------
    .. [1] Korb, Oliver, Thomas Stützle, and Thomas E. Exner.
        “Empirical Scoring Functions for Advanced Protein−Ligand Docking with PLANTS.”
        Journal of Chemical Information and Modeling 49, no. 1 (January 26, 2009): 84–96.
        https://doi.org/10.1021/ci800298z.

    """

    def __init__(
        self,
        ligand: Ligand,
        receptor: Receptor,
        weights: tuple[float, float, float, float, float] | None = None,
        cutoff: float = 8.0,
        k: int = 400,
        ignore_non_polar_hydrogens: bool = True,
    ):
        super().__init__(
            ligand,
            receptor,
            cutoff,
            k,
            ignore_non_polar_hydrogens,
        )
        if weights is None:
            self.weights = (-4.0, -7.0, -0.05, -0.40, 0.50)

        self.__init_plants_interaction_types()
        self.parameters = {
            PlantsPLP._InteractionType.HBOND: (2.3, 2.6, 3.1, 3.4, -4.0, 20.0),
            PlantsPLP._InteractionType.METAL: (1.4, 2.2, 2.6, 2.8, -7.0, 20.0),
            PlantsPLP._InteractionType.BURIED: (3.4, 3.6, 4.5, 5.5, -0.05, 20.0),
            PlantsPLP._InteractionType.NONPOLAR: (3.4, 3.6, 4.5, 5.5, -0.40, 20.0),
            PlantsPLP._InteractionType.REPULSIVE: (3.2, 5.0, 0.05, 10.0),
        }

    class _InteractionType(IntEnum):
        UNKNOWN = 0
        REPULSIVE = 1
        HBOND = 2
        BURIED = 3
        NONPOLAR = 4
        METAL = 5

    def __init_plants_interaction_types(self):
        # construct lut for (ligand_atom, receptor_atom)

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

        lig_types = self.ligand.atom_types[self.lig_mask]
        rec_types = self.receptor.atom_types[self.rec_mask]

        lut = np.zeros(
            (lig_types.shape[0], rec_types.shape[0]),
            dtype=np.int32,
        )

        self.mask_rep = (
            (is_donor(lig_types)[:, None] & is_donor(rec_types)[None, :])
            | (is_acceptor(lig_types)[:, None] & is_acceptor(rec_types)[None, :])
            | (is_donacc(lig_types)[:, None] & is_metal(rec_types)[None, :])
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
                is_donacc(rec_types)
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
        r = r[self.lig_mask]
        idx = idx[self.lig_mask]
        mask = mask[self.lig_mask]
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
