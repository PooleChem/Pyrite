"""`pyrite.io.fix_receptor_pdb`: heterogens go, the protein stays, the input is left alone."""

import os
from pathlib import Path

import pytest

pytest.importorskip("pdbfixer")

from pyrite import Mol
from pyrite.io import fix_receptor_pdb

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "input_files"


@pytest.fixture
def dirty_pdb(tmp_path):
    """A real receptor with a water, a zinc ion and a ligand-like residue appended."""
    lines = [
        line for line in (EXAMPLES / "factor_x.pdb").read_text().splitlines() if line[:4] != "END"
    ]
    lines += [
        "HETATM 9001  O   HOH A9001      10.000  10.000  10.000  1.00 20.00           O",
        "HETATM 9002 ZN    ZN A9002      15.000  15.000  15.000  1.00 20.00          ZN",
        "HETATM 9003  C1  LIG A9003      20.000  20.000  20.000  1.00 20.00           C",
        "END",
    ]
    path = tmp_path / "dirty.pdb"
    path.write_text("\n".join(lines) + "\n")
    return path


def _residues(path):
    return {
        line[17:20].strip()
        for line in Path(path).read_text().splitlines()
        if line[:6] in ("ATOM  ", "HETATM")
    }


def test_heterogens_are_removed_and_the_input_is_untouched(dirty_pdb):
    before = dirty_pdb.read_text()

    fixed = fix_receptor_pdb(str(dirty_pdb))
    try:
        assert not {"HOH", "ZN", "LIG"} & _residues(fixed)
        assert {"ALA", "GLY"} & _residues(fixed)
        assert dirty_pdb.read_text() == before
        assert Mol.from_pdb(fixed).n_atoms > 0
    finally:
        os.unlink(fixed)


def test_water_can_be_kept_and_the_output_path_chosen(dirty_pdb, tmp_path):
    out = tmp_path / "fixed.pdb"

    result = fix_receptor_pdb(dirty_pdb, out_file=str(out), keep_water=True)

    assert result == str(out)
    assert "HOH" in _residues(out)
    assert not {"ZN", "LIG"} & _residues(out)


def _heavy_atoms(path):
    return sum(
        1
        for line in Path(path).read_text().splitlines()
        if line[:4] == "ATOM" and line[76:78].strip() != "H"
    )


def test_a_clean_receptor_keeps_all_its_heavy_atoms(tmp_path):
    original = EXAMPLES / "factor_x.pdb"
    fixed = fix_receptor_pdb(str(original), out_file=str(tmp_path / "fixed.pdb"))

    assert _heavy_atoms(fixed) == _heavy_atoms(original)
