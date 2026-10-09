"""
==============================
Input files (:mod:`pyrite.io`)
==============================

.. currentmodule:: pyrite.io

Reading and preparing input files. Molecules themselves are loaded with
:meth:`Mol.from_pdb <pyrite.Mol.from_pdb>`, :meth:`~pyrite.Mol.from_sdf` and friends.

.. autosummary::
   :toctree: generated/

   fix_receptor_pdb
"""

import os
import tempfile


def fix_receptor_pdb(pdb_file: str, out_file: str | None = None, keep_water: bool = False) -> str:
    """Repair a receptor PDB file so that it can be loaded with :meth:`~pyrite.Mol.from_pdb`.

    Uses `PDBFixer <https://github.com/openmm/pdbfixer>`_ to replace non-standard residues by
    their standard equivalents and to remove heterogens (ligands, ions, and, by default, water).
    The input file is not modified.

    PDBFixer (and OpenMM) are an optional dependency, only needed for this function.

    Parameters
    ----------
    pdb_file : str
        Path to the PDB file of the receptor.
    out_file : str, optional
        Where to write the repaired structure. By default a temporary file is created, which the
        caller should delete (``os.unlink``) after loading it.
    keep_water : bool, default False
        Whether to keep the water molecules.

    Returns
    -------
    str
        The path of the repaired PDB file.

    See Also
    --------
    pyrite.Mol.from_pdb : Load the repaired file.

    Examples
    --------
    >>> import os
    >>> fixed = fix_receptor_pdb("receptor.pdb")
    >>> receptor = Mol.from_pdb(fixed)
    >>> os.unlink(fixed)
    """
    try:
        from openmm.app import PDBFile
        from pdbfixer import PDBFixer
    except ImportError as error:
        raise ImportError(
            "fix_receptor_pdb needs PDBFixer and OpenMM (for instance `conda install -c "
            "conda-forge pdbfixer`)."
        ) from error

    fixer = PDBFixer(filename=os.fspath(pdb_file))
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=keep_water)

    if out_file is None:
        handle, out_file = tempfile.mkstemp(suffix=".pdb")
        os.close(handle)
    with open(out_file, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)
    return out_file
