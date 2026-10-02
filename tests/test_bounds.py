"""Geometry of the `Bounds` shapes: bounding boxes, sampling, and distances."""

import numpy as np
import pytest
from scipy import stats

from pyrite.bounds import CylindricalBounds, RectangularBounds, SphericalBounds

AT = (3.0, -2.0, 5.0)
ROTATION = [0.3, -0.5, 0.8]


def test_sphere_bounding_box_spans_the_radius_on_both_sides():
    bounds = SphericalBounds(7.0, at=AT)

    assert np.allclose(bounds.get_translation_bounds(), [[-4.0, 10.0], [-9.0, 5.0], [-2.0, 12.0]])


def test_cylinder_bounding_box_spans_the_radius_and_the_height():
    bounds = CylindricalBounds(h=10.0, r=6.0, at=AT)

    # The axis is y: the radius spans x and z, the height spans y.
    assert np.allclose(bounds.get_translation_bounds(), [[-3.0, 9.0], [-7.0, 3.0], [-1.0, 11.0]])


@pytest.mark.parametrize("rotation", [None, ROTATION])
def test_sphere_samples_are_uniform_in_the_ball(rotation):
    bounds = SphericalBounds(7.0, at=AT, rotation=rotation)
    positions = bounds.place_random_uniform(4000, rng=np.random.default_rng(0))
    distance = np.linalg.norm(positions - np.array(AT), axis=1)

    assert (distance <= 7.0 + 1e-9).all()
    # Uniform in a ball: (d / r) ** 3 is uniform on [0, 1].
    assert stats.kstest((distance / 7.0) ** 3, "uniform").pvalue > 1e-3
    assert all(bounds.is_within(tuple(p)) for p in positions)


@pytest.mark.parametrize("rotation", [None, ROTATION])
def test_cylinder_samples_are_uniform_in_the_cylinder(rotation):
    bounds = CylindricalBounds(h=10.0, r=6.0, at=AT, rotation=rotation)
    positions = bounds.place_random_uniform(4000, rng=np.random.default_rng(0))
    local = np.array([bounds._world_to_bounds(p) for p in positions])

    assert all(bounds.is_within(tuple(p)) for p in positions)
    # Uniform in a cylinder: the height is uniform on [-h/2, h/2], and (radius / r) ** 2 on [0, 1].
    assert stats.kstest((local[:, 1] + 5.0) / 10.0, "uniform").pvalue > 1e-3
    assert stats.kstest((local[:, 0] ** 2 + local[:, 2] ** 2) / 36.0, "uniform").pvalue > 1e-3


def test_cylinder_squared_distance():
    bounds = CylindricalBounds(h=10.0, r=6.0, at=AT)
    centre = np.array(AT)

    assert bounds.squared_distance(tuple(centre)) == 0.0
    assert bounds.squared_distance(tuple(centre + [6.0, 5.0, 0.0])) == 0.0  # on the rim
    assert bounds.squared_distance(tuple(centre + [8.0, 0.0, 0.0])) == pytest.approx(4.0)  # radial
    assert bounds.squared_distance(tuple(centre + [0.0, -8.0, 0.0])) == pytest.approx(9.0)  # axial
    assert bounds.squared_distance(tuple(centre + [0.0, 8.0, 9.0])) == pytest.approx(9.0 + 9.0)


def test_rectangular_samples_stay_in_the_box_and_cover_it():
    bounds = RectangularBounds((10.0, 8.0, 6.0), at=AT)
    positions = bounds.place_random_uniform(2000, rng=np.random.default_rng(0))

    assert np.allclose(bounds.get_translation_bounds(), [[-2.0, 8.0], [-6.0, 2.0], [2.0, 8.0]])
    assert np.allclose(positions.min(axis=0), [-2.0, -6.0, 2.0], atol=0.05)
    assert np.allclose(positions.max(axis=0), [8.0, 2.0, 8.0], atol=0.05)
