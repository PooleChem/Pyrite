import numpy as np
import math

from pyrite._common import _rotation_matrix_to_euler, _rotation_matrix_from_euler, _translation_matrix_from_coordinates

def test_rotation_matrix_to_euler():
    # identity
    angles = [0,0,0]
    rot_mat = np.eye(3)
    assert np.allclose(angles, _rotation_matrix_to_euler(rot_mat))

    # complex case
    angles = [math.pi / 4, math.pi / 6, math.pi / 3]
    rot_mat = np.array([
                    [0.4330127, -0.43559574, 0.78914913],
                    [0.75,      0.65973961, -0.04736717],
                    [-0.5,      0.61237244, 0.61237244]
                ])

    assert np.allclose(angles, _rotation_matrix_to_euler(rot_mat))


def test_rotation_matrix_from_euler():
    # identity
    angles = [0,0,0]
    rot_mat = np.eye(4)
    assert np.allclose(rot_mat, _rotation_matrix_from_euler(*angles))

    # complex case
    angles = [math.pi / 4, math.pi / 6, math.pi / 3]
    rot_mat = np.array([
                    [0.4330127, -0.43559574, 0.78914913, 0],
                    [0.75,      0.65973961, -0.04736717, 0],
                    [-0.5,      0.61237244, 0.61237244, 0],
                    [0, 0, 0, 1]
                ])

    assert np.allclose(rot_mat, _rotation_matrix_from_euler(*angles))


def test_translation_matrix_from_coordinates():
    # identity
    coords = [0, 0, 0]
    trans_mat = np.eye(4)
    assert np.allclose(trans_mat, _translation_matrix_from_coordinates(*coords))

    # complex case
    coords = [10, -4, 20]
    trans_mat = np.array([[1,0,0,10],[0,1,0,-4],[0,0,1,20],[0,0,0,1]])
    assert np.allclose(trans_mat, _translation_matrix_from_coordinates(*coords))

def test_fix_mol_valence():
    # TODO
    pass


########################
#                      #
#    Molecule class    #
#                      #
########################
# This class is big

from pyrite._common import Mol

def test_create_mol_O2():
    smiles = 'O=O'

    mol = Mol.from_smiles(smiles)
    print(mol.positions)


