import warnings

import numpy as np
from numba import njit
from numpy.typing import NDArray
from rdkit import Chem


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


# A terminal O or N bonded to an atom that has another terminal O or N with the other bond order:
# the two O of a carboxylic acid, the two N of an amidine. RDKit's ``symmetrizeConjugatedTerminalGroups``.
_CONJUGATED_TERMINAL = Chem.MolFromSmarts(
    "[O,N;D1;$([O,N;D1]-[*]=[O,N;D1]),$([O,N;D1]=[*]-[O,N;D1])]~[*]"
)


def _symmetrized_query(rdkit_mol):
    """Return `rdkit_mol` as a query in which conjugated terminal atoms are interchangeable.

    Their bonds match single or double, and the atoms match on the element only (not the charge,
    so the two O of a carboxylate drawn as ``C(=O)[O-]`` are interchangeable too).
    """
    query = Chem.RWMol(rdkit_mol)
    # RDKit stops at 1000 matches by default; a protein has more carboxylates and amidines than that
    for terminal, centre in rdkit_mol.GetSubstructMatches(_CONJUGATED_TERMINAL, maxMatches=10**7):
        bond = query.GetBondBetweenAtoms(terminal, centre)
        if bond.GetBondType() in (Chem.BondType.SINGLE, Chem.BondType.DOUBLE):
            query.ReplaceBond(bond.GetIdx(), Chem.BondFromSmarts("-,="), preserveProps=True)
            element = rdkit_mol.GetAtomWithIdx(terminal).GetAtomicNum()
            query.ReplaceAtom(terminal, Chem.AtomFromSmarts(f"[#{element}]"))
    return query


def _heavy_atoms(mol) -> NDArray:
    """Return the indices of the atoms of `mol` (a :class:`~pyrite.Mol`) that are not hydrogen."""
    return np.array([a.GetIdx() for a in mol.atoms if a.GetAtomicNum() != 1], dtype=np.intp)


def _symmetry_mappings(
    probe, ref, max_matches: int, include_identity: bool = False, heavy_atoms: bool = False
) -> list:
    """Return the atom mappings of `ref` onto `probe` that preserve the molecular graph.

    These are the symmetries of the molecule when `probe` and `ref` are (copies of) the same
    molecule. ``mapping[k]`` is the atom of `probe` that atom ``k`` of `ref` is mapped onto, so
    the pairs ``(mapping[k], k)`` are the ``(probe, ref)`` pairs RDKit's ``CalcRMS`` expects.

    Parameters
    ----------
    probe, ref : Mol
        The molecules to map. `ref` is the query.
    max_matches : int
        The maximum number of mappings. A molecule with many symmetric groups has exponentially
        many; a warning is issued when this limit is reached, as an RMSD over the mappings found
        can then be overestimated.
    include_identity : bool, default False
        Whether to make sure the identity mapping is among the mappings. Only valid when `probe`
        and `ref` have the same atom order. Without it, a truncated list may not contain it.
    heavy_atoms : bool, default False
        Whether to map the heavy atoms only. ``mapping[k]`` is then the atom of `probe` that the
        ``k``-th heavy atom of `ref` (``_heavy_atoms(ref)[k]``) is mapped onto, still as an index
        into `probe`. Hydrogens add many mappings that only swap equivalent hydrogens (3! for
        every methyl group), which can push out the ones that swap heavy atoms.

    Notes
    -----
    The mappings are those RDKit's ``CalcRMS`` uses by default, so an RMSD over them equals it:

    - Chirality is ignored. A mapping of a molecule onto itself can only swap equivalent
      neighbours of an atom, and an atom with equivalent neighbours is no stereocentre; the chiral
      tags are not reliable for this either (perception from 3D tags atoms such as a sulfonyl S or
      a gem-dimethyl C, whose two O or methyl groups must be swappable).
    - The terminal atoms of conjugated groups are equivalent (the two O of a carboxylic acid or
      carboxylate, the two N of an amidine): their bond orders and charges are ignored.
    """
    probe_rdkit, ref_rdkit = probe.rdkit, ref.rdkit
    if heavy_atoms:
        probe_rdkit = Chem.RemoveAllHs(probe_rdkit, sanitize=False)
        ref_rdkit = Chem.RemoveAllHs(ref_rdkit, sanitize=False)
    matches = list(
        probe_rdkit.GetSubstructMatches(
            _symmetrized_query(ref_rdkit), uniquify=False, maxMatches=max_matches
        )
    )
    if len(matches) >= max_matches:
        warnings.warn(
            f"The molecule has at least {max_matches} symmetry-equivalent atom mappings, only "
            "these were used: the RMSD can be overestimated. Raise `max_matches` to include more.",
            stacklevel=3,
        )
    if include_identity:
        identity = tuple(range(probe_rdkit.GetNumAtoms()))
        if identity not in matches:
            matches.insert(0, identity)
    if heavy_atoms:
        # RemoveAllHs keeps the heavy atoms in their order: map back to the indices of `probe`.
        probe_heavy = _heavy_atoms(probe)
        matches = [tuple(int(probe_heavy[i]) for i in m) for m in matches]
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
    """Pack the torsions of a molecule into the arrays `_pose_positions_kernel` works on.

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


@njit(cache=True)
def _rotate_torsions(pos, quads, moving, n_moving, angles):
    """Rotate the atoms beyond every torsion of one conformer by `angles`, in place.

    For torsion ``(a, b, c, d)`` the atoms beyond ``c`` turn by its angle about the axis
    ``b -> c``, taken from the current positions (a nested torsion's axis has been moved by the
    torsions before it). Plain scalar arithmetic: no temporary arrays.
    """
    for t in range(quads.shape[0]):
        b, c = quads[t, 1], quads[t, 2]
        bx, by, bz = pos[b, 0], pos[b, 1], pos[b, 2]
        kx, ky, kz = pos[c, 0] - bx, pos[c, 1] - by, pos[c, 2] - bz
        norm = np.sqrt(kx * kx + ky * ky + kz * kz)
        kx, ky, kz = kx / norm, ky / norm, kz / norm
        co, si = np.cos(angles[t]), np.sin(angles[t])
        one = 1.0 - co
        # Rodrigues' rotation about the unit axis k, as a matrix
        r00, r01, r02 = co + kx * kx * one, kx * ky * one - kz * si, kx * kz * one + ky * si
        r10, r11, r12 = ky * kx * one + kz * si, co + ky * ky * one, ky * kz * one - kx * si
        r20, r21, r22 = kz * kx * one - ky * si, kz * ky * one + kx * si, co + kz * kz * one
        for m in range(n_moving[t]):
            j = moving[t, m]
            px, py, pz = pos[j, 0] - bx, pos[j, 1] - by, pos[j, 2] - bz
            pos[j, 0] = bx + r00 * px + r01 * py + r02 * pz
            pos[j, 1] = by + r10 * px + r11 * py + r12 * pz
            pos[j, 2] = bz + r20 * px + r21 * py + r22 * pz


@njit(cache=True)
def _pose_positions_kernel(values, centered, euler, n_rot, quads, moving, n_moving, offsets):
    """The atom positions of a batch of poses. See `Mol.pose_to_positions`.

    Rotate the reference geometry (`centered`: the center atom at the origin) about the center
    atom, place the center atom at the translation, then turn every torsion by its target minus
    its reference value (`offsets`). Closed form: a rigid motion does not change a dihedral, and
    turning one torsion does not change another's, so the dihedral to correct is always the
    reference one, and nothing needs measuring.
    """
    n, n_atoms, n_tors = values.shape[0], centered.shape[0], offsets.shape[0]
    out = np.empty((n, n_atoms, 3))
    angles = np.empty(n_tors)
    for i in range(n):
        if euler:  # (roll, pitch, yaw), R = Rz(yaw) Ry(pitch) Rx(roll)
            cr, sr = np.cos(values[i, 0]), np.sin(values[i, 0])
            cp, sp = np.cos(values[i, 1]), np.sin(values[i, 1])
            cy, sy = np.cos(values[i, 2]), np.sin(values[i, 2])
            r00, r01, r02 = cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr
            r10, r11, r12 = sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr
            r20, r21, r22 = -sp, cp * sr, cp * cr
        else:  # (w, x, y, z), normalised here as SciPy does
            w, x, y, z = values[i, 0], values[i, 1], values[i, 2], values[i, 3]
            q = np.sqrt(w * w + x * x + y * y + z * z)
            w, x, y, z = w / q, x / q, y / q, z / q
            r00, r01, r02 = 1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)
            r10, r11, r12 = 2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)
            r20, r21, r22 = 2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)
        tx, ty, tz = values[i, n_rot], values[i, n_rot + 1], values[i, n_rot + 2]
        for j in range(n_atoms):
            x0, y0, z0 = centered[j, 0], centered[j, 1], centered[j, 2]
            out[i, j, 0] = r00 * x0 + r01 * y0 + r02 * z0 + tx
            out[i, j, 1] = r10 * x0 + r11 * y0 + r12 * z0 + ty
            out[i, j, 2] = r20 * x0 + r21 * y0 + r22 * z0 + tz
        for t in range(n_tors):
            angles[t] = values[i, n_rot + 3 + t] - offsets[t]
        _rotate_torsions(out[i], quads, moving, n_moving, angles)
    return out


@njit(cache=True)
def _pose_gradient_kernel(positions, forces, values, euler, n_rot, center, quads, moving, n_moving):
    """The gradient of a score with respect to a pose, from the posed positions and the forces.

    `forces` is the gradient of the score with respect to the atom positions (``dS/dx_j``,
    shape ``(n_atoms, 3)``). Every pose variable moves the atoms rigidly, so its derivative is a
    sum over the atoms it moves: a force for the translation, a torque for a rotation.

    - translation: ``sum_j F_j``;
    - rotation: the torque ``tau = sum_j (x_j - x_center) x F_j``, projected on the axis each
      rotation variable turns about (euler), or mapped through the derivative of the normalised
      quaternion (quat);
    - torsion ``(a, b, c, d)``: ``u . sum_moved (x_j - x_b) x F_j``, with ``u`` the unit axis
      ``b -> c`` in the posed geometry. Turning one torsion never changes another's dihedral, so
      a nested torsion's derivative is this, too.
    """
    n_dims = values.shape[0]
    gradient = np.zeros(n_dims)
    cx, cy, cz = positions[center, 0], positions[center, 1], positions[center, 2]
    fx = fy = fz = 0.0
    tx = ty = tz = 0.0
    for j in range(positions.shape[0]):
        f0, f1, f2 = forces[j, 0], forces[j, 1], forces[j, 2]
        fx += f0
        fy += f1
        fz += f2
        rx, ry, rz = positions[j, 0] - cx, positions[j, 1] - cy, positions[j, 2] - cz
        tx += ry * f2 - rz * f1
        ty += rz * f0 - rx * f2
        tz += rx * f1 - ry * f0
    gradient[n_rot] = fx
    gradient[n_rot + 1] = fy
    gradient[n_rot + 2] = fz
    if euler:  # the axes of roll, pitch and yaw: Rz(yaw) Ry(pitch) e_x, Rz(yaw) e_y, e_z
        cp, sp = np.cos(values[1]), np.sin(values[1])
        cyw, syw = np.cos(values[2]), np.sin(values[2])
        gradient[0] = cyw * cp * tx + syw * cp * ty - sp * tz
        gradient[1] = -syw * tx + cyw * ty
        gradient[2] = tz
    else:
        # For the normalised q = (w, v): omega = 2 (w dv - dw v + v x dv), so the gradient is
        # (-2 v . tau, 2 (w tau + tau x v)). It is orthogonal to q already, so the derivative of
        # the normalisation only divides it by |q|.
        w, x, y, z = values[0], values[1], values[2], values[3]
        norm = np.sqrt(w * w + x * x + y * y + z * z)
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
        gradient[0] = -2.0 * (x * tx + y * ty + z * tz) / norm
        gradient[1] = 2.0 * (w * tx + ty * z - tz * y) / norm
        gradient[2] = 2.0 * (w * ty + tz * x - tx * z) / norm
        gradient[3] = 2.0 * (w * tz + tx * y - ty * x) / norm
    for t in range(quads.shape[0]):
        b, c = quads[t, 1], quads[t, 2]
        bx, by, bz = positions[b, 0], positions[b, 1], positions[b, 2]
        ux, uy, uz = positions[c, 0] - bx, positions[c, 1] - by, positions[c, 2] - bz
        norm = np.sqrt(ux * ux + uy * uy + uz * uz)
        ux, uy, uz = ux / norm, uy / norm, uz / norm
        sx = sy = sz = 0.0
        for m in range(n_moving[t]):
            j = moving[t, m]
            rx, ry, rz = positions[j, 0] - bx, positions[j, 1] - by, positions[j, 2] - bz
            f0, f1, f2 = forces[j, 0], forces[j, 1], forces[j, 2]
            sx += ry * f2 - rz * f1
            sy += rz * f0 - rx * f2
            sz += rx * f1 - ry * f0
        gradient[n_rot + 3 + t] = ux * sx + uy * sy + uz * sz
    return gradient
