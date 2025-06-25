"""

====================================================
Bounds module (:mod:`pyrite.bounds`)
====================================================

.. currentmodule:: pyrite.bounds

Tools for defining :class:`Bounds` objects, which can be used for placement, scoring, and as bounds
for an optimizer.


.. autosummary::
   :toctree: generated/

   Bounds


Simple bounds
-------------

.. autosummary::
   :toctree: generated/

   RectangularBounds
   SphericalBounds
   CylindricalBounds


Pocket
------

.. autosummary::
   :toctree: generated/

   Pocket


See Also
--------
:ref:`bounds_scoring_functions`
    Scoring functions that make use of ``Bounds``.


"""

import warnings
from abc import ABC, abstractmethod
from collections import deque

import numpy as np
from numpy.typing import NDArray, ArrayLike
from rdkit.Geometry import Point3D
from scipy.spatial import KDTree

from pyrite._common import _rotation_matrix, Ligand, Receptor


class Bounds(ABC):
    """Bounds class responsible for representing and managing 3D ligand bounds.

    The class provides mechanisms for defining a bounding box with specified dimensions,
    position, and rotation. It supports operations such as determining if a point lies
    within bounds, transforming coordinates between local and world spaces, and computing
    translation bounds. The class is designed to be extended by subclasses for cases
    requiring additional, specific functionality.

    .. note::
        This is an :class:`~abc.ABC` (`abstract base class`) and should be subclasses. Please refer to
        the Notes section for more information on how to do this.


    Parameters
    ----------
    bbv : array_like
        A 3 element vector pointing to one of the corners of the bounding box, if it were to be
        centered at the origin, with its axes aligned with the world coordinate axes.
    at : array_like, default [0,0,0]
        The location of the bounding box's center. Defaults to world origin.
    rotation : array_like, default [0,0,0]
        The rotation of the bounding box in Euler angles or a 3x3 rotation matrix. Defaults to
        no rotation.


    """

    # Bounding box centered at the origin. Smallest box that contains the whole bounds.
    # Vector to corner
    # _bounding_box_at_origin: NDArray
    # __bounding_box_rotated: NDArray
    #
    # _at: NDArray = []
    # _rotation_matrix = []  # Rotation matrix from Bound coordinates to World coordinates
    # __inv_rotation_matrix = (
    #     []
    # )  # Inverse rotation matrix. From world coordinates to bound coordinates

    __default_draw_options = {"size": (400, 300), "wireframe": True, "color": "gold"}

    def __init__(
        self,
        bbv: tuple[float, float, float],
        at: tuple[float, float, float] = None,
        rotation: NDArray = None,
    ):
        if at is None:
            at = [0, 0, 0]
        if rotation is None:
            rotation = [0, 0, 0]

        self._bounding_box_at_origin = np.array(bbv)
        self._at = np.array(at)

        rotation = np.array(rotation)
        if rotation.shape == (3,):
            # Euler angles
            self._rotation_matrix = np.array(_rotation_matrix(*rotation)[:3, :3])
        elif rotation.shape == (3, 3):
            # Rotation matrix
            self._rotation_matrix = np.array(rotation)
        else:
            raise TypeError("Invalid rotation argument.")

        self.__inv_rotation_matrix = np.linalg.inv(self._rotation_matrix)

        self.__compute_rotated_boundingbox()

        self.draw_options = self.__default_draw_options.copy()

    def __compute_rotated_boundingbox(self) -> None:
        box = np.array(
            [
                self._bounding_box_at_origin,
                -self._bounding_box_at_origin,
                [
                    self._bounding_box_at_origin[0],
                    self._bounding_box_at_origin[1],
                    -self._bounding_box_at_origin[2],
                ],
                [
                    self._bounding_box_at_origin[0],
                    -self._bounding_box_at_origin[1],
                    -self._bounding_box_at_origin[2],
                ],
                [
                    self._bounding_box_at_origin[0],
                    -self._bounding_box_at_origin[1],
                    self._bounding_box_at_origin[2],
                ],
                [
                    -self._bounding_box_at_origin[0],
                    self._bounding_box_at_origin[1],
                    -self._bounding_box_at_origin[2],
                ],
                [
                    -self._bounding_box_at_origin[0],
                    self._bounding_box_at_origin[1],
                    self._bounding_box_at_origin[2],
                ],
                [
                    -self._bounding_box_at_origin[0],
                    -self._bounding_box_at_origin[1],
                    -self._bounding_box_at_origin[2],
                ],
            ]
        )

        rotated_box = np.matmul(box, self._rotation_matrix)

        self.__bounding_box_rotated = np.array(
            [max(rotated_box[:, 0]), max(rotated_box[:, 1]), max(rotated_box[:, 2])]
        )

    def get_translation_bounds(self) -> list[tuple[float, float]]:
        """Returns the translation bounds of the ``Bounds`` object.

        This method determines the minimum and maximum coordinates for the translation
        bounds within the bounding box, considering its current alignment and rotation.


        Returns
        -------
        list[tuple[float, float]]
            A list of tuples where each tuple represents the minimum and
            maximum bounds along a coordinate axis.

        """
        return list(
            zip(
                -self.__bounding_box_rotated + self._at,
                self.__bounding_box_rotated + self._at,
            )
        )

    def is_within(self, p: tuple[float, float, float]) -> bool:
        """Checks if a given point lies within the bounding box of the object.


        Parameters
        ----------
        p : array_like
            The point to be checked.

        Returns
        -------
        bool

        """

        p = self._world_to_bounds(p)

        return (
            (abs(p[0]) <= self._bounding_box_at_origin[0])
            and (abs(p[1]) <= self._bounding_box_at_origin[1])
            and (abs(p[2]) <= self._bounding_box_at_origin[2])
        )

    # TODO: get rid of squared distance
    @abstractmethod
    def squared_distance(self, p: tuple[float, float, float]) -> float:
        """Calculate the squared distance of a point to the ``Bounds`` object.

        This method computes the squared Euclidean distance between the current bounds
        and a given point `p`.

        Parameters
        ----------
        p : array_like
            The point to be compared with the bounds.

        Returns
        -------
        float

        """

    def distance(self, p: tuple[float, float, float]) -> float:
        """Calculate the distance between a given point and the object.

        This method computes the Euclidean distance between the current bounds and a given
        point `p`.

        Parameters
        ----------
        p : array_like
            The point to be compared with the bounds.

        Returns
        -------
        float

        """
        return np.sqrt(self.squared_distance(p))

    @abstractmethod
    def transform_sample_to_bounds(self, v: NDArray) -> NDArray:
        """Transforms a sample point to the specified bounds.

        This method maps a sample point `v`, with shape ``(3,)`` or a list of sample points `v`,
        with shape ``(3,n)`` onto the target bounds, ensuring that the transformation adheres to
        the specific domain requirements defined within the ``Bounds``. The exact behavior of
        the transformation must be implemented in a subclass.

        Parameters
        ----------
        v : array_like
            A vector of shape ``(3,)`` or ``(3,n)`` of values within [0,1) representing the sample
            point(s) to transform.

        Returns
        -------
        numpy.ndarray
            An array with a shape equal to `v` representing the transformed sample points.
        """

    @staticmethod
    def transform_sample_to_2pi(v: NDArray) -> NDArray:
        """Transform the input array values to the range of [-π, π).

        Parameters
        ----------
        v : numpy.ndarray
            A vector of shape ``(3,)`` or ``(3,n)`` of values within [0,1) representing the sample
            point(s) to transform.

        Returns
        -------
        numpy.ndarray
            An array with a shape equal to `v` representing the transformed sample points.

        """
        v_2d = np.atleast_2d(v)

        v_2d[:, :] *= 2 * np.pi
        v_2d[:, :] -= np.pi

        if v_2d.ndim < 2:
            return v_2d[0, :]

        return v_2d

    def _world_to_bounds(self, v: NDArray) -> NDArray:
        return np.matmul(np.array(v) - self._at, self.__inv_rotation_matrix)

    def _bounds_to_world(self, v: NDArray) -> NDArray:
        return np.matmul(np.array(v), self._rotation_matrix) + self._at

    def place_random_uniform(self, n: int = 1) -> NDArray:
        """Generates an array of random samples within specified bounds.

        Parameters
        ----------
        n : int, default 1
            Number of samples to generate.


        Returns
        -------
        numpy.ndarray
            A NumPy array containing the generated random samples. The array has
            a shape of ``(n, 6)``, where the first three columns represent rotation and the
            last three columns represent translation.

        """
        s = np.random.rand(n, 3 + 3)

        s[:, 0:3] = Bounds.transform_sample_to_2pi(s[:, 0:3])
        s[:, 3:6] = self.transform_sample_to_bounds(s[:, 3:6])

        return s

    def place_grid(self, n: int = 1, rotation_grid: bool = False) -> NDArray:
        """Generate a grid of `n` samples representing rotation and translation, and convert
        them into the required bounds.

        Parameters
        ----------
        n : int
            Number of points to discretize each axis in the grid.
        rotation_grid : bool, default False
            Whether to create a discretized rotation grid (True),
            or use random rotation values (default, False).

        Returns
        -------
        numpy.ndarray
            A numpy array representing the grid of sampled transformations, with
            rotation angles in radians and translations in their corresponding adjusted
            bounds.

        """

        rotation_grid = [np.linspace(0, 1, n)] * 3 if rotation_grid else [0.0] * 3
        translation_grid = [np.linspace(0, 1, n)] * 3

        axes = rotation_grid + translation_grid

        roll, yaw, pitch, x, y, z = np.meshgrid(*axes, indexing="ij")
        s = np.vstack(
            [roll.ravel(), yaw.ravel(), pitch.ravel(), x.ravel(), y.ravel(), z.ravel()]
        ).T

        if not rotation_grid:
            s[:, 0:3] = np.random.rand(s.shape[0], 3)

        s[:, 0:3] = Bounds.transform_sample_to_2pi(s[:, 0:3])
        s[:, 3:6] = self.transform_sample_to_bounds(s[:, 3:6])

        return s

    def _viewer_add_(self, viewer, c_m_id, options: dict = None):
        if options is None:
            options = {}

        l_options = self.draw_options.copy()
        l_options.update(options)

        viewer.view.addBox(
            {
                "center": {"x": self._at[0], "y": self._at[1], "z": self._at[2]},
                "dimensions": {
                    "w": self._bounding_box_at_origin[0] * 2,
                    "h": self._bounding_box_at_origin[1] * 2,
                    "d": self._bounding_box_at_origin[2] * 2,
                },
                "color": l_options["color"],
                "wireframe": l_options["wireframe"],
            }
        )

        return c_m_id

    def get_bounds(
        self,
        ligand: Ligand,
        angle_bounds: tuple[float, float] = (-2 * np.pi, 2 * np.pi),
        dihedral_bounds: tuple[float, float] = (-2 * np.pi, 2 * np.pi),
    ):
        """Returns the translation bounds of the ``Bounds`` object.

        This method determines the minimum and maximum coordinates for the translation
        bounds within the bounding box, considering its current alignment and rotation.

        Parameters
        ----------
        ligand : Ligand
            The ligand to use to retrieve the dihedral bounds.
        angle_bounds : tuple[float, float], default (-2 * np.pi, 2 * np.pi)
            The bounds to use for the rotation angles.
        dihedral_bounds : tuple[float, float], default (-2 * np.pi, 2 * np.pi)
            The bounds to use for the dihedral angles.

        Returns
        -------
        list[tuple[float, float]]
            A list of tuples where each tuple represents the minimum and
            maximum bounds along a coordinate axis.

        """
        bounds = [angle_bounds] * 6 + [dihedral_bounds] * len(ligand.dihedral_angles)
        bounds[3:6] = self.get_translation_bounds()

        return bounds


class RectangularBounds(Bounds):
    """Represents a rectangular bounding box in a 3D coordinate system.


    Parameters
    ----------
    s : array_like, float
        The size of the rectangle in the ``x``, ``y``, and ``z`` axes. If a single float is
        supplied, a cube with equal side length `s` is created.
    at : array_like, default [0,0,0]
        The location of the rectangles center. Defaults to world origin.
    rotation : array_like, default [0,0,0]
        The rotation of the rectangle in Euler angles or a 3x3 rotation matrix. Defaults to
        no rotation.


    """

    def __init__(
        self,
        s: tuple[float, float, float] | float,
        at: tuple[float, float, float] = None,
        rotation: NDArray = None,
    ):
        if at is None:
            at = [0, 0, 0]
        if rotation is None:
            rotation = [0, 0, 0]
        if isinstance(s, (int, float)):
            s = (float(s),) * 3

        super().__init__((s[0] / 2, s[1] / 2, s[2] / 2), at, rotation)

    # TODO: consider rotation.
    @classmethod
    def from_ligand(cls, ligand: Ligand, padding: float = 0.0):
        """Creates an instance of the class based on the given ligand and optional padding.

        .. warning::
            The created bounding box is always aligned to the world axes.

        Parameters
        ----------
        ligand : Ligand
            The ``Ligand`` to autobox.
        padding : float, optional
            Optional padding around the ligand.


        Returns
        -------
        RectangularBounds


        """
        conf = ligand.GetConformer()
        points = np.empty((ligand.GetNumAtoms(), 3))
        for i, atom in enumerate(ligand.GetAtoms()):
            ai = atom.GetIdx()
            points[i] = conf.GetAtomPosition(ai)

        dimensions = np.max(points, axis=0) - np.min(points, axis=0) + 2 * padding
        center = np.mean([np.min(points, axis=0), np.max(points, axis=0)], axis=0)
        return cls(dimensions, center)

    def squared_distance(self, p: tuple[float, float, float]) -> float:
        p = self._world_to_bounds(np.array(p))

        diff = [0, 0, 0]
        for axis in range(3):
            if abs(p[axis]) >= self._bounding_box_at_origin[axis]:
                diff[axis] = abs(p[axis]) - self._bounding_box_at_origin[axis]

        return np.sum(np.square(diff))

    def transform_sample_to_bounds(self, v: NDArray) -> NDArray:
        v *= 2 * self._bounding_box_at_origin
        v -= self._bounding_box_at_origin

        return self._bounds_to_world(v)


class SphericalBounds(Bounds):
    """Represents a spherical bounding 'box' in a 3D coordinate system.


    Parameters
    ----------
    r : float
        Radius of the sphere.
    at : array_like, default [0,0,0]
        The location of the rectangles center. Defaults to world origin.
    rotation : array_like, default [0,0,0]
        The rotation of the rectangle in Euler angles or a 3x3 rotation matrix. Defaults to
        no rotation.

    """

    r = 0
    __r2 = 0

    def __init__(
        self, r, at: tuple[float, float, float] = None, rotation: NDArray = None
    ):
        if at is None:
            at = [0, 0, 0]
        if rotation is None:
            rotation = [0, 0, 0]

        super().__init__([r / 2, r / 2, r / 2], at, rotation)
        self.r = r
        self.__r2 = r * r

    def is_within(self, p):
        return super().is_within(p) and self.squared_distance(p) <= 0.0

    def squared_distance(self, p):
        return max(0.0, np.sum(np.square(self._world_to_bounds(p))) - self.__r2)

    # Spherical coordinates
    def transform_sample_to_bounds(self, v: NDArray):
        v_2d = np.atleast_2d(v)

        phi = v_2d[:, 0] * 2 * np.pi  # [0, 2pi]
        costheta = v_2d[:, 1] * 2 - 1  # [-1, 1]
        u = v_2d[:, 2]  # [0, 1]

        theta = np.arccos(costheta)
        r_ = self.r * np.cbrt(u)
        # R = self.r * u

        v_2d[:, 0] = r_ * np.sin(theta) * np.cos(phi)
        v_2d[:, 1] = r_ * np.sin(theta) * np.sin(phi)
        v_2d[:, 2] = r_ * np.cos(theta)

        v_2d = self._bounds_to_world(v_2d)

        if v.ndim < 2:
            return v_2d[0, :]

        return v_2d


class CylindricalBounds(Bounds):
    """Represents a cylindrical bounding 'box' in a 3D coordinate system.


    Parameters
    ----------
    h : float
        The height of the cylinder.
    r : float
        The radius of the cylinder.
    at : array_like, default [0,0,0]
        The location of the rectangles center. Defaults to world origin.
    rotation : array_like, default [0,0,0]
        The rotation of the rectangle in Euler angles or a 3x3 rotation matrix. Defaults to
        no rotation.

    """

    r = 0
    __r2 = 0

    def __init__(
        self, h, r, at: tuple[float, float, float] = None, rotation: NDArray = None
    ):
        if at is None:
            at = [0, 0, 0]
        if rotation is None:
            rotation = [0, 0, 0]

        super().__init__([r / 2, h / 2, r / 2], at, rotation)
        self.r = r
        self.__r2 = r * r

    def is_within(self, p):
        return super().is_within(p) and self._squared_distance_to_axis(p) <= self.__r2

    def _squared_distance_to_axis(self, p):
        return np.sum(np.square(self._world_to_bounds(p)[[0, 2]]))

    def squared_distance(self, p):
        p = np.subtract(p, self._at)

        dh = max(0.0, abs(p[1]) - self._bounding_box_at_origin[1])
        dr = max(0.0, self._squared_distance_to_axis(p) - self.__r2)

        return max(0.0, dh**2 + dr**2)

    def transform_sample_to_bounds(self, v: NDArray):
        v_2d = np.atleast_2d(v)

        phi = v_2d[:, 0] * 2 * np.pi  # [0, 2pi] (angle)
        h = v_2d[:, 1]  # [0, 1] (h)
        u = v_2d[:, 2]  # [0, 1] (r)

        r_ = self.r * np.sqrt(u)

        v_2d[:, 0] = r_ * np.cos(phi)
        v_2d[:, 1] = (
            h * self._bounding_box_at_origin[1] - self._bounding_box_at_origin[1] / 2
        )
        v_2d[:, 2] = r_ * -np.sin(phi)

        if v.ndim < 2:
            return v_2d[0, :]

        return v_2d


class Pocket(Bounds):
    """Represents a binding pocket formed of alpha spheres.

    .. warning::
        When creating a ``Pocket`` based scoring function, try not to use the :meth:`distance`
        method. This method is non-vectorized and therefore slow.

        Instead use a :class:`~pyrite.scoring.dependencies.KNNDependency` on `centers`.

    Parameters
    ----------
    centers : array_like
        The centers of the alpha spheres.
    radii : array_like
        The radii of the alpha spheres.
    charges : array_like, default None
        The charges of the alpha spheres. This can be used to assign weights to specific alpha
        spheres in a :class:`ScoringFunction`. Defaults to zeros.


    """

    __default_draw_options = {
        "size": (400, 300),
        "wireframe": True,
        "color": "gold",
        "opacity": 1,
    }

    def __init__(
        self,
        centers: NDArray,
        radii: NDArray,
        charges: NDArray = None,
    ):
        self.centers = centers
        self.radii = radii
        self.charges = charges if charges is not None else np.zeros(len(centers))

        # if kdtree:
        #     self.tree = KDTree(self.centers)
        #     self.sum_of_distances = self._sum_of_distances_kdtree
        # else:
        #     self.sum_of_distances = self.__sum_of_distances_vectorized

        bbv, at = self.__compute_bounding_box()
        super().__init__(bbv, at=at, rotation=[0, 0, 0])
        self.draw_options = self.__default_draw_options.copy()

    @classmethod
    def from_receptor(
        cls,
        receptor: Receptor,
        grid_size: float = 0.8,
        sphere_radius: float = 1.4,
        neighbors: int = 18,
        weight_residues: ArrayLike = None,
        leq_distance_to_protein: int = 4,
        gt_distance_to_outside: int = 8,
        leq_distance_to_residue: int = 2,
        weight_cap: int = 20,
        solvent_accessible: bool = True,
    ):
        """Creates a ``Pocket`` from a :class:`~pyrite.Receptor`.

        This method creates a grid, and fills the space with alpha spheres. These spheres are
        optionally assigned a weight based on their distance to certain residues. The closer to the
        residue, the lower their weight.

        Furthermore, BFS is used to calculate the distance to the protein and to an outside shell.
        This allows for filtering of the alpha spheres, and selection of a specific pocket depth.

        The distances and weights are Manhattan distance. I.e., the distance in grid-point
        hops ("city blocks"). What counts a a distance of one can be tweaked, as shown in the
        plot below. Here, every cube represents a grid point. If `neighbors` is set to 6, only
        yellow points are considered as neighbors, with distance ``1``. If `neighbors` is set to
        18, the green points are also considered as distance ``1``. With `neighbors` set to 26, the
        blue points are also considered neighbors.

        .. plot::
           :width: 80%
           :alt: Neighbors example

           import matplotlib.pyplot as plt
           from mpl_toolkits.mplot3d.art3d import Poly3DCollection
           from itertools import product

           def plot_cube(ax, origin, size, color):
               x, y, z = origin
               # Define the 8 vertices of the cube
               vertices = [
                   [x, y, z],
                   [x + size, y, z],
                   [x + size, y + size, z],
                   [x, y + size, z],
                   [x, y, z + size],
                   [x + size, y, z + size],
                   [x + size, y + size, z + size],
                   [x, y + size, z + size]
               ]
               # Define the 6 faces by listing the vertices that make up each face
               faces = [
                   [vertices[0], vertices[1], vertices[2], vertices[3]],
                   [vertices[4], vertices[5], vertices[6], vertices[7]],
                   [vertices[0], vertices[1], vertices[5], vertices[4]],
                   [vertices[2], vertices[3], vertices[7], vertices[6]],
                   [vertices[1], vertices[2], vertices[6], vertices[5]],
                   [vertices[4], vertices[7], vertices[3], vertices[0]]
               ]
               # Create a 3D polygon collection for the cube
               poly3d = Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors='k', alpha=0.6)
               ax.add_collection3d(poly3d)

           # Create a 3D plot
           fig = plt.figure()
           ax = fig.add_subplot(111, projection='3d')

           # Plot the central cube at (0,0,0)
           plot_cube(ax, (0, 0, 0), 1, 'red')

           # Plot neighbors with different connectivity colors
           for dx, dy, dz in product([-1, 0, 1], repeat=3):
               if dx == 0 and dy == 0 and dz == 0:
                   continue
               # Determine connectivity type by counting non-zero offsets
               nonzero_count = sum(1 for v in (dx, dy, dz) if v != 0)
               if nonzero_count == 1:
                   color = 'yellow'   # 6-connectivity (face neighbors)
               elif nonzero_count == 2:
                   color = 'green'    # 18-connectivity (edge neighbors)
               else:
                   color = 'blue'     # 26-connectivity (corner neighbors)
               plot_cube(ax, (dx, dy, dz), 1, color)

           # Set plot limits and labels
           ax.set_xlim(-1, 2)
           ax.set_ylim(-1, 2)
           ax.set_zlim(-1, 2)
           ax.set_xlabel('X')
           ax.set_ylabel('Y')
           ax.set_zlabel('Z')

           ax.set_xticks([])
           ax.set_yticks([])
           ax.set_zticks([])


        Parameters
        ----------
        receptor : Receptor
            The receptor to use.
        grid_size : float, default 0.8
            The grid size to use.
        sphere_radius : float, default 1.4
            The radius of the alpha spheres.
        neighbors : {6, 18, 26}, default 18
            The number of grid-neighbors to consider.
        weight_residues : ArrayLike, optional
            An optional array of residues, used for the assignment of weights to the alpha spheres.
            The residues should be supplied by name and id, e.g. ``TYR228``.
        leq_distance_to_protein : int, default 4
            A filter for the distance to the protein. Spheres that are further away will be removed.
        gt_distance_to_outside : int, default 8
            A filter for the distance to the outside shell.
            Spheres that are closer to the outside will be removed.
        leq_distance_to_residue : int, default 2
            A filter for the distance to the residue. Spheres that are further away will be removed.
        weight_cap : int, default 20
            The maximum weight. Any value higher than this will be se to this.
        solvent_accessible : bool, default True
            Whether to only include solvent-accessible spheres. If `solvent_accessible` is
            ``False``, internal cavities will be included.


        Returns
        -------
        Pocket
        """
        tree = KDTree(receptor.positions)

        padding = 10 * grid_size
        x_min = np.min(receptor.positions[:, 0]) - padding
        x_max = np.max(receptor.positions[:, 0]) + padding
        y_min = np.min(receptor.positions[:, 1]) - padding
        y_max = np.max(receptor.positions[:, 1]) + padding
        z_min = np.min(receptor.positions[:, 2]) - padding
        z_max = np.max(receptor.positions[:, 2]) + padding

        x_grid = np.arange(x_min, x_max, grid_size)
        y_grid = np.arange(y_min, y_max, grid_size)
        z_grid = np.arange(z_min, z_max, grid_size)

        x, y, z = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")
        grid = np.stack((x, y, z), axis=-1)

        if neighbors not in {6, 18, 26}:
            raise ValueError(
                f"Invalid number of neighbors: {neighbors}. Must be one of {{6, 18, 26}}."
            )
        max_abs = 1 if neighbors == 6 else (2 if neighbors == 18 else 3)
        offsets = [
            (dx, dy, dz)
            for dx in (-1, 0, +1)
            for dy in (-1, 0, +1)
            for dz in (-1, 0, +1)
            if not (dx == dy == dz == 0) and ((abs(dx) + abs(dy) + abs(dz)) <= max_abs)
        ]

        i_min, j_min, k_min = 0, 0, 0
        i_max, j_max, k_max = len(x_grid) - 1, len(y_grid) - 1, len(z_grid) - 1

        def grid_neighbors(i, j, k):
            # FOR grid point (I, J, K)
            # retrieve the indices of the neighbors.
            for dx, dy, dz in offsets:
                xx, yy, zz = i + dx, j + dy, k + dz
                if (
                    (xx >= i_min)
                    and (xx <= i_max)
                    and (yy >= j_min)
                    and (yy <= j_max)
                    and (zz >= k_min)
                    and (zz <= k_max)
                ):
                    yield xx, yy, zz

        result_close = tree.query_ball_point(
            grid, 2 * sphere_radius, return_length=True
        )

        occupied = result_close >= 1

        # Set distance_to_protein
        distance_to_protein = -np.ones(occupied.shape, dtype=int)

        q = deque()

        for idx in np.ndindex(occupied.shape):
            if occupied[idx]:
                for nbr in grid_neighbors(*idx):
                    if not occupied[nbr]:
                        q.append(nbr)
                        distance_to_protein[nbr] = 1

        # BFS for distance_to_protein
        while q:
            idx = q.popleft()
            for nbr in grid_neighbors(*idx):
                if not occupied[nbr]:
                    if distance_to_protein[nbr] < 0:
                        distance_to_protein[nbr] = distance_to_protein[idx] + 1
                        q.append(nbr)

        if weight_residues is not None:
            # Set distance_to_residue

            distance_to_residue_occ = -np.ones(occupied.shape, dtype=int)
            distance_to_residue = -np.ones(occupied.shape, dtype=int)

            q_occ = deque()
            q = deque()

            # Find closest non occupied points.
            for idx in np.ndindex(occupied.shape):
                if occupied[idx]:
                    pt_idx = tree.query(grid[idx], k=1)[1]
                    residue = receptor.residues[pt_idx]
                    if residue in weight_residues:
                        for nbr in grid_neighbors(*idx):
                            if not occupied[nbr]:
                                q.append(nbr)
                                distance_to_residue[nbr] = 1
                            else:
                                q_occ.append(nbr)
                                distance_to_residue_occ[nbr] = 1

            occ_distance_limit = float("inf")
            if not q:
                while q_occ:
                    idx = q_occ.popleft()
                    # Stop condition
                    if distance_to_residue_occ[idx] > occ_distance_limit:
                        break

                    for nbr in grid_neighbors(*idx):
                        if not occupied[nbr]:
                            occ_distance_limit = distance_to_residue_occ[idx]
                            distance_to_residue[nbr] = 1
                            q.append(nbr)
                        elif distance_to_residue_occ[nbr] < 0:
                            distance_to_residue_occ[nbr] = (
                                distance_to_residue_occ[idx] + 1
                            )
                            q_occ.append(nbr)

            # BFS distance_to_residue

            while q:
                idx = q.popleft()
                for nbr in grid_neighbors(*idx):
                    # if not occupied[nbr]:
                    if distance_to_residue[nbr] < 0:
                        distance_to_residue[nbr] = distance_to_residue[idx] + 1
                        q.append(nbr)

            distance_to_residue[occupied] = -1.0

        # Distance_to_outside
        distance_to_outside = -np.ones(occupied.shape, dtype=int)

        for idx in np.ndindex(occupied.shape):
            if (
                distance_to_protein[idx] == 10
                or idx[0] == i_min
                or idx[0] == i_max
                or idx[1] == j_min
                or idx[1] == j_max
                or idx[2] == k_min
                or idx[2] == k_max
            ):
                q.append(idx)
                distance_to_outside[idx] = 0

        while q:
            idx = q.popleft()
            for nbr in grid_neighbors(*idx):
                if not occupied[nbr] and distance_to_outside[nbr] < 0:
                    distance_to_outside[nbr] = distance_to_outside[idx] + 1
                    q.append(nbr)

        accessible = distance_to_outside >= 0

        mask = (distance_to_protein <= leq_distance_to_protein) & (
            distance_to_outside > gt_distance_to_outside
        )
        if weight_residues is not None:
            mask &= distance_to_residue <= leq_distance_to_residue
        if solvent_accessible:
            mask &= accessible

        centers = grid[mask].reshape(-1, 3)
        radii = np.ones(len(centers)) * sphere_radius
        charges = np.zeros(len(centers))
        if weight_residues is not None:
            clamped_distance_to_residue = np.minimum(distance_to_residue, weight_cap)
            charges = clamped_distance_to_residue[mask].reshape(-1)
            # charges = charges.max() - charges

        return cls(centers, radii, charges)

    @classmethod
    def from_pqr(cls, pqr_file: str):
        """Loads a ``Pocket`` from a PQR file.

        The centers of the spheres, their radii and charges are read from the PQR file.

        Parameters
        ----------
        pqr_file : str
            The path of the PQR file to read.

        Returns
        -------
        Pocket
        """
        centers, radii, charges = cls.__parse_pqr(pqr_file)
        return cls(centers, radii, charges)

    def to_pqr(
        self,
        pqr_file: str,
    ):
        """Write the ``Pocket`` to a PQR file.

        Parameters
        ----------
        pqr_file : str
            The path of the PQR file to write to.

        """
        atom_name = "SPH"
        res_name = "SPH"
        chain_id = "A"
        start_residue_number = 1

        with open(pqr_file, "w") as f:
            f.write("REMARK   Generated by write_spheres_to_pqr\n")
            for i, ((x, y, z), r, c) in enumerate(
                zip(self.centers, self.radii, self.charges), start=1
            ):
                res_seq = start_residue_number + i
                # PQR format columns (PDB-like):
                # 1-6  Record name, 7-11 Atom serial, 13-16 Atom name, 17    altLoc,
                # 18-20 Residue name, 22 Chain ID, 23-26 Residue seq, 27    iCode,
                # 31-38 x, 39-46 y, 47-54 z, 55-62 charge, 63-70 radius
                line = (
                    f"ATOM  {i:5d} {atom_name:^4s} {res_name:>3s} {chain_id:1s}"
                    f"{res_seq:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    f"{c:8.4f}{r:8.4f}\n"
                )
                f.write(line)

    @staticmethod
    def __parse_pqr(pqr_file: str):
        centers = []
        radii = []
        charges = []

        with open(pqr_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("ATOM"):
                    line = line.split()
                    centers.append(
                        np.array([float(line[5]), float(line[6]), float(line[7])])
                    )
                    charges.append(float(line[8]))
                    radii.append(float(line[9]))

        return np.array(centers), np.array(radii), np.array(charges)

    def __compute_bounding_box(self):
        # x_max, y_max, z_max = float("-inf"), float("-inf"), float("-inf")
        # x_min, y_min, z_min = float("inf"), float("inf"), float("inf")
        #
        # for i, c in enumerate(self.centers):
        #     x_max = max(x_max, c[0] + self.radii[i])
        #     x_min = min(x_min, c[0] - self.radii[i])
        #     y_max = max(y_max, c[1] + self.radii[i])
        #     y_min = min(y_min, c[1] - self.radii[i])
        #     z_max = max(z_max, c[2] + self.radii[i])
        #     z_min = min(z_min, c[2] - self.radii[i])
        #
        # print(x_max, y_max, z_max)
        # print(x_min, y_min, z_min)
        #
        # at = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2])
        # bbv = np.array([x_max, y_max, z_max]) - at

        dimensions = (
            np.max(self.centers, axis=0)
            - np.min(self.centers, axis=0)
            + 2 * self.radii[0]
        )
        center = np.mean(
            [np.min(self.centers, axis=0), np.max(self.centers, axis=0)], axis=0
        )

        # print(center, dimensions)

        return tuple((dimensions, center))

    def is_within(self, p: tuple[float, float, float]) -> bool:
        return super().is_within(p) and np.isclose(self.distance(p), 0.0)

    def squared_distance(self, p: tuple[float, float, float]) -> float:
        return self.distance(p) ** 2

    def distance(self, p: tuple[float, float, float]) -> float:
        probe_point = Point3D(*p)

        minimum_distance = float("inf")
        for i, c in enumerate(self.centers):
            if minimum_distance <= 0:  # Point is inside a pocket
                break
            minimum_distance = min(
                minimum_distance, probe_point.Distance(Point3D(*c)) - self.radii[i]
            )
        return max(0.0, minimum_distance)

    def __sum_of_distances_vectorized(self, p: NDArray):
        warnings.deprecated()
        diff = p[:, None, :] - self.centers[None, :, :]  # (N, M, 3)
        dist = np.sqrt(np.sum(diff * diff, axis=2))  # (N, M)

        dist_to_surface = dist - self.radii[None, :]
        dist_to_surface_clamped = np.maximum(dist_to_surface, 0.0)

        return float(dist_to_surface_clamped.min(axis=1).sum())

    def _sum_of_distances_kdtree(self, p: NDArray):
        warnings.deprecated()
        nearest_neighbor_distances = self.tree.query(p, k=1)[0]
        return float(
            np.sum(np.maximum(nearest_neighbor_distances - self.radii[0], 0.0))
        )

    def transform_sample_to_bounds(self, v: NDArray) -> NDArray:
        raise NotImplementedError

    def intersect(self, bounds: Bounds, padding: float = 0.0):
        """Intersect the ``Pocket`` with the given bounds.

        For every sphere in the ``Pocket``, checks whether any part of it is within `padding`
        of the `bounds`. All spheres that are fully outside of `bounds` + `padding` are removed.

        Parameters
        ----------
        bounds : Bounds
            The bounds to intersect with.
        padding : float
            Padding around the bounds.

        Returns
        -------
        Pocket
        """
        i_to_remove = []
        for i, c in enumerate(self.centers):
            if bounds.distance(c) - self.radii[i] > padding:
                i_to_remove.append(i)

        for i in reversed(i_to_remove):
            self.centers = np.delete(self.centers, i, axis=0)
            self.radii = np.delete(self.radii, i)

        bbv, at = self.__compute_bounding_box()
        self._bounding_box_at_origin = np.array(bbv)
        self._at = at

        return self

    def place_random_uniform(self, n: int = 1) -> NDArray:
        bounding_box = RectangularBounds(self._bounding_box_at_origin, self._at)

        tree = KDTree(self.centers)

        results = []

        while len(results) < n:
            # Generate 4 * (n - len) samples
            s = np.random.rand(max(40, 4 * (n - len(results))), 3 + 3)
            s[:, 0:3] = Bounds.transform_sample_to_2pi(s[:, 0:3])
            s[:, 3:6] = bounding_box.transform_sample_to_bounds(s[:, 3:6])

            # Check if the points are inside the pocket
            nearest_neighbor_distances = tree.query(s[:, 3:], k=1)[0]
            valid = (nearest_neighbor_distances - self.radii[0]) <= 0.0

            results.extend(s[valid])

        return np.array(results[:n])

    def _viewer_add_(self, viewer, c_m_id, options: dict = None):
        if options is None:
            options = {}

        l_options = self.draw_options.copy()
        l_options.update(options)

        for i, c in enumerate(self.centers):
            viewer.view.addSphere(
                {
                    "center": {"x": c[0], "y": c[1], "z": c[2]},
                    "radius": self.radii[i],
                    "color": l_options["color"],
                    "wireframe": l_options["wireframe"],
                    "quality": 0,
                    "opacity": l_options["opacity"],
                }
            )

        return c_m_id
