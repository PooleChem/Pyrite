import warnings

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


def _symmetry_mappings(probe, ref, max_matches: int, include_identity: bool = False) -> list:
    """Return the atom mappings of `ref` onto `probe` that preserve the molecular graph.

    These are the symmetries of the molecule when `probe` and `ref` are (copies of) the same
    molecule. ``mapping[k]`` is the atom of `probe` that atom ``k`` of `ref` is mapped onto, so
    the pairs ``(mapping[k], k)`` are the ``(probe, ref)`` pairs RDKit's ``CalcRMS`` expects.

    Parameters
    ----------
    probe, ref : rdkit.Chem.Mol
        The molecules to map. `ref` is the query.
    max_matches : int
        The maximum number of mappings. A molecule with many symmetric groups has exponentially
        many; a warning is issued when this limit is reached, as an RMSD over the mappings found
        can then be overestimated.
    include_identity : bool, default False
        Whether to make sure the identity mapping is among the mappings. Only valid when `probe`
        and `ref` have the same atom order. Without it, a truncated list may not contain it.
    """
    matches = list(
        probe.rdkit.GetSubstructMatches(
            ref.rdkit, uniquify=False, useChirality=True, maxMatches=max_matches
        )
    )
    if len(matches) >= max_matches:
        warnings.warn(
            f"The molecule has at least {max_matches} symmetry-equivalent atom mappings, only "
            "these were used: the RMSD can be overestimated. Raise `max_matches` to include more.",
            stacklevel=3,
        )
    if include_identity:
        identity = tuple(range(probe.n_atoms))
        if identity not in matches:
            matches.insert(0, identity)
    return matches


def _atoms_beyond(adjacency: list, start: int, blocked: int) -> NDArray:
    """Return the atoms reachable from `start` without passing the bond `start`-`blocked`.

    For a torsion about the bond ``b-c`` these are the atoms that move when ``c`` is the start.
    The bond must not be in a ring (a rotatable bond never is).
    """
    seen, stack = {start}, [start]
    while stack:
        i = stack.pop()
        for j in adjacency[i]:
            if j not in seen and not (i == start and j == blocked):
                seen.add(j)
                stack.append(j)
    if blocked in seen:
        raise ValueError(f"The bond {start}-{blocked} is in a ring and cannot be rotated.")
    return np.array(sorted(seen), dtype=np.intp)


def _pack_torsions(quads: list, moving: list) -> tuple[NDArray, NDArray, NDArray]:
    """Pack the torsions of a molecule into the arrays `_apply_torsions` works on.

    Parameters
    ----------
    quads : list of 4-tuples
        The atoms ``(a, b, c, d)`` of every torsion.
    moving : list of ndarray
        For every torsion, the indices of the atoms it moves.

    Returns
    -------
    quads : ndarray
        Shape ``(n_torsions, 4)``.
    moving : ndarray
        Shape ``(n_torsions, width)``: the moved atoms of each torsion, padded with zeros.
    n_moving : ndarray
        Shape ``(n_torsions,)``: how many entries of every row of `moving` are real.
    """
    n_moving = np.array([len(m) for m in moving], dtype=np.int64)
    padded = np.zeros((len(moving), max(int(n_moving.max(initial=0)), 1)), dtype=np.int64)
    for t, indices in enumerate(moving):
        padded[t, : len(indices)] = indices
    return np.array(quads, dtype=np.int64).reshape(-1, 4), padded, n_moving


@njit
def _apply_torsions_kernel(positions, quads, moving, n_moving, torsions):
    """Set the torsions of every conformer in `positions`, in place. See `_apply_torsions`."""
    for i in range(positions.shape[0]):
        pos = positions[i]
        for t in range(quads.shape[0]):
            a, b, c, d = quads[t, 0], quads[t, 1], quads[t, 2], quads[t, 3]
            # the current dihedral, with RDKit's sign convention
            b0 = pos[a] - pos[b]
            axis = pos[c] - pos[b]
            axis = axis / np.sqrt(np.sum(axis * axis))
            b2 = pos[d] - pos[c]
            v = b0 - np.sum(b0 * axis) * axis
            w = b2 - np.sum(b2 * axis) * axis
            cross = np.array(
                [
                    axis[1] * v[2] - axis[2] * v[1],
                    axis[2] * v[0] - axis[0] * v[2],
                    axis[0] * v[1] - axis[1] * v[0],
                ]
            )
            angle = torsions[i, t] - np.arctan2(np.sum(cross * w), np.sum(v * w))
            cos, sin = np.cos(angle), np.sin(angle)
            # rotate the atoms beyond c about the axis through b (Rodrigues)
            origin = pos[b].copy()
            for m in range(n_moving[t]):
                p = pos[moving[t, m]] - origin
                k_cross_p = np.array(
                    [
                        axis[1] * p[2] - axis[2] * p[1],
                        axis[2] * p[0] - axis[0] * p[2],
                        axis[0] * p[1] - axis[1] * p[0],
                    ]
                )
                along = np.sum(axis * p)
                pos[moving[t, m]] = origin + p * cos + k_cross_p * sin + axis * along * (1.0 - cos)


def _apply_torsions(positions: NDArray, packed: tuple, torsions: NDArray) -> NDArray:
    """Set the torsion angles of a batch of conformers.

    Equivalent to RDKit's ``SetDihedralRad`` for every torsion in turn: the current dihedral of
    ``(a, b, c, d)`` is measured, and the atoms beyond ``c`` are rotated about the bond ``b-c`` by
    the difference to the target. The torsions are absolute, and rotating about one never changes
    another's dihedral, so the result does not depend on the order.

    Parameters
    ----------
    positions : ndarray
        Shape ``(n, n_atoms, 3)``. Not modified.
    packed : tuple
        The torsions of the molecule, from `_pack_torsions`.
    torsions : ndarray
        Shape ``(n, n_torsions)``, the target angles in radians.

    Returns
    -------
    ndarray
        The new positions, shape ``(n, n_atoms, 3)``.
    """
    out = np.array(positions, dtype=np.float64, order="C")
    _apply_torsions_kernel(out, *packed, np.ascontiguousarray(torsions, dtype=np.float64))
    return out
