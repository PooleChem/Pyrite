import numpy as np
from numba import njit
from numpy.typing import NDArray

def _rotation_matrix_to_euler(r: NDArray):
    """
    Extract ZYX (yaw-pitch-roll) Euler angles from a 3×3 rotation matrix R.
    Returns (phi, theta, psi) = (roll, pitch, yaw) in radians.
    """
    # # clamp to handle numerical errors outside [-1,1]
    sy = -r[2, 0]
    theta = np.arcsin(np.clip(sy, -1.0, 1.0))

    # Check for gimbal lock
    if np.isclose(np.cos(theta), 0.0):
        # Gimbal lock: pitch is ±90°
        # Roll and yaw are coupled; set roll=0 and compute yaw:
        phi = 0.0
        psi = np.arctan2(-r[0, 1], r[1, 1])
    else:
        phi = np.arctan2(r[2, 1], r[2, 2])  # roll
        psi = np.arctan2(r[1, 0], r[0, 0])  # yaw

    return phi, theta, psi


# TODO: Support Quaternions
@njit
def _rotation_matrix_from_euler(roll: float, pitch: float, yaw: float):
    r"""Create a rotation matrix from Euler angles.

    Create a 4x4 rotation matrix from Euler angles (roll, pitch, yaw) in radians.


    Parameters
    ----------
    roll : float
        Roll angle in radians.

    pitch : float
        Pitch angle in radians.

    yaw : float
        Yaw angle in radians.

    Returns
    -------
    rotation_matrix : NDArray
        A rotation matrix of shape (4, 4) representing the rotation transformation for the specified Euler angles.
    """
    sin_roll = np.sin(roll)
    cos_roll = np.cos(roll)
    sin_pitch = np.sin(pitch)
    cos_pitch = np.cos(pitch)
    sin_yaw = np.sin(yaw)
    cos_yaw = np.cos(yaw)

    m = np.eye(4)
    m[0, 0] = cos_yaw * cos_pitch
    m[0, 1] = cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll
    m[0, 2] = cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll
    m[1, 0] = sin_yaw * cos_pitch
    m[1, 1] = sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll
    m[1, 2] = sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll
    m[2, 0] = -sin_pitch
    m[2, 1] = cos_pitch * sin_roll
    m[2, 2] = cos_pitch * cos_roll
    return m


@njit
def _rotation_matrix_from_quat(qw: float, qx: float, qy: float, qz: float):
    r"""Create a rotation matrix from a quaternion.

    Create a 4x4 rotation matrix from a quaternion ``(qw, qx, qy, qz)``. The
    quaternion does not need to be pre-normalized — the general (homogeneous)
    conversion formula is used, which divides by the squared norm and is
    numerically equivalent to normalizing the quaternion first, without
    needing a `sqrt`.

    Parameters
    ----------
    qw : float
        Scalar (real) component of the quaternion.

    qx : float
        First vector (imaginary) component of the quaternion.

    qy : float
        Second vector (imaginary) component of the quaternion.

    qz : float
        Third vector (imaginary) component of the quaternion.

    Returns
    -------
    rotation_matrix : NDArray
        A rotation matrix of shape (4, 4) representing the rotation transformation for the specified quaternion.
    """
    n = qw * qw + qx * qx + qy * qy + qz * qz
    s = 2.0 / n

    wx = s * qw * qx
    wy = s * qw * qy
    wz = s * qw * qz
    xx = s * qx * qx
    xy = s * qx * qy
    xz = s * qx * qz
    yy = s * qy * qy
    yz = s * qy * qz
    zz = s * qz * qz

    m = np.eye(4)
    m[0, 0] = 1.0 - (yy + zz)
    m[0, 1] = xy - wz
    m[0, 2] = xz + wy
    m[1, 0] = xy + wz
    m[1, 1] = 1.0 - (xx + zz)
    m[1, 2] = yz - wx
    m[2, 0] = xz - wy
    m[2, 1] = yz + wx
    m[2, 2] = 1.0 - (xx + yy)
    return m


@njit
def _translation_matrix_from_coordinates(x: float, y: float, z: float):
    """Create a translation matrix from coordinates.

    Create a 4x4 translation matrix from the provided relative x, y, and z coordinates.

    Parameters
    ----------
    x, y, z : float
        Coordinates for the translation.

    Returns
    -------
    translation_matrix : NDArray
    """
    m = np.eye(4)
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m
