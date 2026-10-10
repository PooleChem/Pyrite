"""Measure the speed of every scoring function, and grade it for the docs.

Every term is timed on the 2BOH complex of the user guide, with its default parameters, on poses
near the crystal pose:

- single: the time of one ``get_score`` call, one pose at a time (what a local search pays), over
  the same poses as the batch: scoring one pose over and over is up to 2.5x faster, as the CPU
  caches keep what the last call used;
- batched: the time per pose through ``batch_scores``, in batches of ``BATCH`` poses.

The grade is the order of magnitude of the time per evaluation, one emoji per factor of 10 (see
``GRADES``). Run
from the repository root::

    python benchmarks/speed_grades.py
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pyrite import Mol, Poses  # noqa: E402
from pyrite import scoring as s  # noqa: E402
from pyrite.bounds import Pocket, RectangularBounds  # noqa: E402
from pyrite.io import fix_receptor_pdb  # noqa: E402
from pyrite.scoring.grid import GridScore  # noqa: E402

# (upper bound of the time per evaluation in microseconds, emoji): one emoji per factor of 10
GRADES = [(1, "🚀"), (10, "✈️"), (100, "🚗"), (1_000, "🚲"), (10_000, "🐢"), (np.inf, "🐌")]
BATCH = 256
MIN_TIME = 0.5  # seconds of timing per measurement


def grade(microseconds: float) -> str:
    return next(emoji for bound, emoji in GRADES if microseconds <= bound)


def time_per_evaluation(f, n_per_call: int) -> float:
    """The time of one call of ``f``, divided by ``n_per_call``, in microseconds."""
    f()  # warm up: numba compilation, caches
    calls, start = 0, time.perf_counter()
    while (elapsed := time.perf_counter() - start) < MIN_TIME:
        f()
        calls += 1
    return elapsed / (calls * n_per_call) * 1e6


def main():
    warnings.simplefilter("ignore")
    data = ROOT / "doc/source/user_guide/input"
    receptor = Mol.from_pdb(fix_receptor_pdb(str(data / "2boh.pdb")), hydrogens="add")
    ligand = Mol.from_sdf(str(data / "2boh_ligand.sdf"), flexible=True, hydrogens="add")
    box = RectangularBounds.autobox(ligand, padding=1.0)
    pocket = Pocket.from_mol(receptor).intersect(box, padding=2.0)

    rng = np.random.default_rng(0)
    pose = ligand.input_pose
    poses = Poses.from_parts(
        np.tile(pose.rotation, (BATCH, 1)) + rng.normal(0.0, 0.2, (BATCH, 3)),
        pose.translation + rng.normal(0.0, 0.5, (BATCH, 3)),
        pose.torsions + rng.normal(0.0, 0.3, (BATCH, ligand.n_tors)),
    )

    vina_terms = (
        s.Gaussian(ligand, receptor)
        + s.Gaussian(ligand, receptor, offset=3.0, width=2.0)
        + s.Repulsion(ligand, receptor)
        + s.Hydrophobic(ligand, receptor)
        + s.NonDirHBond(ligand, receptor)
    )
    terms = {
        "NumAtoms": s.NumAtoms(ligand),
        "NumTors": s.NumTors(ligand),
        "InternalOverlap": s.InternalOverlap(ligand),
        "InternalEnergy": s.InternalEnergy(ligand),
        "RMSD": s.RMSD(ligand),
        "Crowding": s.Crowding(ligand, register_initial=True),
        "DistanceToPocket": s.DistanceToPocket(ligand, pocket),
        "WeightedBoundsOverlap": s.WeightedBoundsOverlap(ligand, pocket),
        "OutOfBoundsPenalty": s.OutOfBoundsPenalty(ligand, box),
        "NumProteinAtomsWithinA": s.NumProteinAtomsWithinA(ligand, receptor),
        "Gaussian": s.Gaussian(ligand, receptor),
        "Repulsion": s.Repulsion(ligand, receptor),
        "Hydrophobic": s.Hydrophobic(ligand, receptor),
        "NonHydrophobic": s.NonHydrophobic(ligand, receptor),
        "NonDirHBond": s.NonDirHBond(ligand, receptor),
        "LJ": s.LJ(ligand, receptor),
        "VDW": s.VDW(ligand, receptor),
        "NonDirHBondLJ": s.NonDirHBondLJ(ligand, receptor),
        "ElectroStatic": s.ElectroStatic(ligand, receptor),
        "AD4Solvation": s.AD4Solvation(ligand, receptor),
        "PlantsPLP": s.PlantsPLP(ligand, receptor),
        "GridScore": GridScore(vina_terms, box),
        "vina_like": s.vina_like(ligand, receptor),
    }

    print(f"{'term':24s} {'single (us)':>12s}     {'batched (us)':>12s}")
    for name, term in terms.items():
        single = time_per_evaluation(lambda term=term: [term.get_score(p) for p in poses], BATCH)
        batched = time_per_evaluation(lambda term=term: term.batch_scores(poses), BATCH)
        print(
            f"{name:24s} {single:12.1f} {grade(single)}   {batched:12.1f} {grade(batched)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
