"""Shared fixtures, and the registries of implementations that ``test_contracts.py`` checks.

Adding a new implementation
---------------------------
Every implementation of an extension point is run through the conformance checks of
``contracts.py`` automatically, once it is registered below:

- a **scoring function** or a **Bounds**: add one line to ``SCORING_FACTORIES`` or
  ``BOUNDS_FACTORIES``, mapping the class to a function that builds an instance from the shared
  context ``c`` (``c.ligand``, ``c.receptor``, ``c.site``, ``c.pocket``). A public subclass that
  is not registered makes ``test_every_*_is_registered`` fail, so it cannot be forgotten.
- a **hop**, **temperature schedule** or **step size rule**: these are plain callables without a
  base class, so they cannot be discovered; add a line to ``HOP_FACTORIES``,
  ``SCHEDULE_FACTORIES`` or ``STEPSIZE_FACTORIES``.

A factory should build a *fresh, cheap* instance with default-ish arguments. Instances are cached
for the session, so a conformance check must not modify the implementation it checks.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyrite import Mol, PoseLayout
from pyrite.bounds import CylindricalBounds, Pocket, RectangularBounds, SphericalBounds
from pyrite.scoring import (
    LJ,
    RMSD,
    VDW,
    AD4Solvation,
    Clamp,
    ConstantTerm,
    Crowding,
    DistanceToPocket,
    ElectroStatic,
    Gaussian,
    Hydrophobic,
    InternalEnergy,
    InternalOverlap,
    NonDirHBond,
    NonDirHBondLJ,
    NonHydrophobic,
    NumAtoms,
    NumProteinAtomsWithinA,
    NumTors,
    OutOfBoundsPenalty,
    PlantsPLP,
    Repulsion,
    WeightedBoundsOverlap,
)
from pyrite.scoring.grid import GridScore
from pyrite.search import adaptive_stepsize, geometric_annealing, place_in, random_hop

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "input_files"

# The layouts every hop and every Bounds is checked for: euler, quaternion, and a rigid molecule.
LAYOUTS = [PoseLayout("euler", 4), PoseLayout("quat", 2), PoseLayout("euler", 0)]


def _center(c) -> tuple[float, float, float]:
    return tuple(float(x) for x in c.ligand.get_positions().mean(axis=0))


# ---------------------------------------------------------------------------
# Registries: one factory per implementation, ``factory(context) -> instance``
# ---------------------------------------------------------------------------

SCORING_FACTORIES = {
    Clamp: lambda c: Clamp(Gaussian(c.ligand, c.receptor), -0.5, 0.5),
    ConstantTerm: lambda c: ConstantTerm(2.5),
    DistanceToPocket: lambda c: DistanceToPocket(c.ligand, c.pocket),
    OutOfBoundsPenalty: lambda c: OutOfBoundsPenalty(c.ligand, c.site),
    WeightedBoundsOverlap: lambda c: WeightedBoundsOverlap(c.ligand, c.pocket),
    GridScore: lambda c: GridScore(
        Gaussian(c.ligand, c.receptor) + Repulsion(c.ligand, c.receptor),
        c.site,
        spacing=1.0,
    ),
    InternalEnergy: lambda c: InternalEnergy(c.ligand),
    InternalOverlap: lambda c: InternalOverlap(c.ligand),
    Crowding: lambda c: Crowding(c.ligand),
    NumAtoms: lambda c: NumAtoms(c.ligand),
    NumProteinAtomsWithinA: lambda c: NumProteinAtomsWithinA(c.ligand, c.receptor),
    NumTors: lambda c: NumTors(c.ligand),
    RMSD: lambda c: RMSD(c.ligand),
    AD4Solvation: lambda c: AD4Solvation(c.ligand, c.receptor),
    ElectroStatic: lambda c: ElectroStatic(c.ligand, c.receptor),
    Gaussian: lambda c: Gaussian(c.ligand, c.receptor),
    Hydrophobic: lambda c: Hydrophobic(c.ligand, c.receptor),
    LJ: lambda c: LJ(c.ligand, c.receptor),
    NonDirHBond: lambda c: NonDirHBond(c.ligand, c.receptor),
    NonDirHBondLJ: lambda c: NonDirHBondLJ(c.ligand, c.receptor),
    NonHydrophobic: lambda c: NonHydrophobic(c.ligand, c.receptor),
    PlantsPLP: lambda c: PlantsPLP(c.ligand, c.receptor),
    Repulsion: lambda c: Repulsion(c.ligand, c.receptor),
    VDW: lambda c: VDW(c.ligand, c.receptor),
}

BOUNDS_FACTORIES = {
    CylindricalBounds: lambda c: CylindricalBounds(h=10.0, r=6.0, at=_center(c)),
    Pocket: lambda c: c.pocket,
    RectangularBounds: lambda c: RectangularBounds((10.0, 8.0, 6.0), at=_center(c)),
    SphericalBounds: lambda c: SphericalBounds(7.0, at=_center(c)),
}

HOP_FACTORIES = {
    "random_hop": lambda c: random_hop(c.site.get_translation_bounds()),
    "random_hop (no bounds)": lambda c: random_hop(),
}

SCHEDULE_FACTORIES = {
    "geometric_annealing": lambda c: geometric_annealing(2.0, 0.1),
    "constant": lambda c: lambda i, niter: 1.0,
}

STEPSIZE_FACTORIES = {
    "adaptive_stepsize": lambda c: adaptive_stepsize(),
}


def pytest_generate_tests(metafunc):
    """Parametrize the tests that ask for an implementation by one of the registries."""
    classes = {"scoring_class": SCORING_FACTORIES, "bounds_class": BOUNDS_FACTORIES}
    names = {
        "hop_name": HOP_FACTORIES,
        "schedule_name": SCHEDULE_FACTORIES,
        "stepsize_rule_name": STEPSIZE_FACTORIES,
    }
    for argument, registry in classes.items():
        if argument in metafunc.fixturenames:
            metafunc.parametrize(argument, list(registry), ids=[cls.__name__ for cls in registry])
    for argument, registry in names.items():
        if argument in metafunc.fixturenames:
            metafunc.parametrize(argument, list(registry), ids=list(registry))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ctx() -> SimpleNamespace:
    """The shared context: the ``factor_x`` ligand and receptor, a binding site, a pocket, poses."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ligand = Mol.from_sdf(str(EXAMPLES / "factor_x_ligand.sdf"), flexible=True)
        receptor = Mol.from_pdb(str(EXAMPLES / "factor_x.pdb"))
        site = RectangularBounds.autobox(ligand, padding=1.0)
        pocket = Pocket.from_mol(
            receptor, gt_distance_to_outside=10, solvent_accessible=False
        ).intersect(site, padding=2.0)
    poses = place_in(ligand, site, 4, 4, rng=np.random.default_rng(0))
    # A partner to combine every scoring function with: nearest-neighbour terms with different
    # k and cutoffs that share one query, so that merging them is exercised.
    partner = (
        Gaussian(ligand, receptor, k=400)
        + 0.5 * Repulsion(ligand, receptor, k=100)
        + Hydrophobic(ligand, receptor, k=50)
    )
    return SimpleNamespace(
        ligand=ligand, receptor=receptor, site=site, pocket=pocket, poses=poses, partner=partner
    )


@pytest.fixture(scope="session")
def instances() -> dict:
    """A cache of the instances built by the factories, shared by the whole session."""
    return {}


@pytest.fixture
def scoring_function(scoring_class, ctx, instances):
    key = ("scoring", scoring_class)
    if key not in instances:
        instances[key] = SCORING_FACTORIES[scoring_class](ctx)
    return instances[key]


@pytest.fixture
def bounds(bounds_class, ctx, instances):
    key = ("bounds", bounds_class)
    if key not in instances:
        instances[key] = BOUNDS_FACTORIES[bounds_class](ctx)
    return instances[key]


@pytest.fixture
def scoring_registry() -> dict:
    return SCORING_FACTORIES


@pytest.fixture
def bounds_registry() -> dict:
    return BOUNDS_FACTORIES
