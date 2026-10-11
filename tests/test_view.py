"""The viewer is plain HTML: it must work without a kernel (static docs, exported notebooks)."""

import html

import numpy as np
import pytest

from pyrite import Mol, Poses, Viewer
from pyrite.bounds import RectangularBounds

pytest.importorskip("py3Dmol")


@pytest.fixture
def ligand():
    return Mol.from_smiles("CCCCc1ccccc1O", flexible=True)


@pytest.fixture
def poses(ligand):
    poses = Poses.from_list([ligand.input_pose] * 5)
    poses.torsions[:] = np.random.default_rng(0).uniform(-np.pi, np.pi, poses.torsions.shape)
    return poses


def _page(viewer):
    out = viewer._repr_html_()
    assert out.startswith("<iframe srcdoc=")
    return html.unescape(out)


def test_slider_has_every_pose_as_a_frame(ligand, poses):
    page = _page(Viewer(width=300, height=200).add_v(ligand, poses))
    assert "addModelsAsFrames" in page
    assert page.count("$$$$") == len(poses)
    assert f'max="{len(poses) - 1}"' in page
    assert f"Pose 1 / {len(poses)}" in page


def test_without_slider_every_pose_is_a_model(ligand, poses):
    page = _page(Viewer().add_v(ligand, poses, slider=False))
    assert 'type="range"' not in page
    assert page.count(".addModel(") == len(poses)


def test_other_objects_and_mol_display(ligand):
    page = _page(Viewer(ligand, RectangularBounds.autobox(ligand, padding=1.0)))
    assert "addBox" in page and 'type="range"' not in page
    assert ligand._repr_html_().startswith("<iframe srcdoc=")


def test_viewer_leaves_the_molecule_untouched(ligand, poses):
    n_conformers = ligand.rdkit.GetNumConformers()
    positions = ligand.get_positions().copy()
    Viewer().add_v(ligand, poses)._repr_html_()
    assert ligand.rdkit.GetNumConformers() == n_conformers
    np.testing.assert_array_equal(ligand.get_positions(), positions)


def test_zoom_defaults_to_the_poses(ligand, poses):
    other = Mol.from_smiles("CCO")
    viewer = Viewer(other).add_v(ligand, poses)
    assert "const zoom = [1]" in _page(viewer)
    assert "const zoom = [0]" in _page(viewer.zoom_to(other))
    assert "const zoom = []" in _page(viewer.zoom_to())
    assert "const zoom = []" in _page(Viewer(other))


def test_zoom_to_unknown_object_raises(ligand):
    with pytest.raises(ValueError, match="not a molecule in this viewer"):
        Viewer().zoom_to(ligand)


def test_stickresidues_by_number_and_name():
    from pathlib import Path

    receptor = Mol.from_pdb(str(Path(__file__).parents[1] / "examples/input_files/factor_x.pdb"))
    receptor.set_draw_options({"stickresidues": [57, "HIS"]})
    page = _page(Viewer(receptor))
    assert '"resi": 57' in page or '"resi":57' in page
    assert '"resn": "HIS"' in page or '"resn":"HIS"' in page
