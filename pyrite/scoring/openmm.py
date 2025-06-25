import itertools

import numpy as np
import tqdm
from openff.toolkit import Molecule, Topology, Quantity
from openff.units.openmm import from_openmm
from openmm import (
    LangevinIntegrator,
    HarmonicAngleForce,
    HarmonicBondForce,
    PeriodicTorsionForce,
    CMMotionRemover,
    NonbondedForce,
    CustomGBForce,
)
from openmm import Platform
from openmm import app as openmm
from openmm import unit
from openmmforcefields.generators import SystemGenerator
from rdkit import Chem

from ._base import ScoringFunction
from .._common import Ligand


class OpenMMForceField(ScoringFunction):
    """ """

    # TODO: remove class attributes
    _complex: Topology  #
    _complex_rd: Chem.Mol  # RDkit

    __complex_ligand_id: int
    __ligand_pos: Quantity
    __complex_pos: list

    __ligand_indices: set
    __protein_indices: set

    __forcefield_kwargs = {
        "constraints": openmm.HBonds,
        "nonbondedCutoff": 1 * unit.nanometer,
    }
    __system = None
    __integrator = None
    __simulation: openmm.Simulation
    __platform: Platform

    def __init__(
        self,
        ligand: Ligand | Chem.Mol,
        receptor: Topology,
        minimize_receptor: int = 4,
    ):
        super().__init__(ligand, receptor)

        self.__ligand_indices = set()
        self.__protein_indices = set()

        # TODO:
        # Change ligand residue numbers voor warning
        res_info = Chem.AtomPDBResidueInfo()
        res_info.SetResidueNumber(1001)
        [a.SetPDBResidueInfo(res_info) for a in self.ligand.GetAtoms()]

        # res_info = Chem.AtomPDBResidueInfo()
        # res_info.SetResidueNumber(1002)
        # [a.SetPDBResidueInfo(res_info) for a in self.non_dock_ligands[0].GetAtoms()]

        self.__integrator = LangevinIntegrator(
            300 * unit.kelvin, 1 / unit.picosecond, 1 * unit.femtoseconds
        )

        self.__create_complex()
        molecules = [
            Molecule.from_rdkit(
                self.ligand, allow_undefined_stereo=True, hydrogens_are_explicit=True
            )
        ]  # + [Molecule.from_rdkit(m) for m in non_dock_ligands]
        system_generator = SystemGenerator(
            forcefields=["amber14-all.xml", "implicit/obc2.xml"],
            small_molecule_forcefield="openff-2.0.0",
            molecules=molecules,
            forcefield_kwargs=self.__forcefield_kwargs,
            periodic_forcefield_kwargs={"nonbondedMethod": openmm.CutoffNonPeriodic},
            cache="./out/ff-cache.json",
        )

        self.__platform = Platform.getPlatform("OpenCL")

        self.__system = system_generator.create_system(self._complex.to_openmm())
        # self._prune_forces()

        self.__simulation = openmm.Simulation(
            self._complex.to_openmm(), self.__system, self.__integrator, self.__platform
        )
        self.__simulation.context.setPositions(
            self._complex.get_positions().to_openmm()
        )

        if minimize_receptor > 0:
            for _ in tqdm(range(minimize_receptor), desc="Minimizing receptor"):
                # self.__simulation.step(100)
                self.__simulation.minimizeEnergy()

            # state = self.__simulation.context.getState(energy=True)
            # energy = state.getPotentialEnergy() / unit.kilocalories_per_mole
            # print(energy)

            # TODO: update ligand positions?
            #  (does not really matter in this case, but might be good for different optimization methods)
            #  Should we do this? That would also change bond angles and lengths, which would introduce bias
            #  towards crystal docked pose -> Data leakage.
            self.__update_complex()

            with open("out/minimized.pdb", "w") as f:
                openmm.PDBFile.writeFile(
                    self._complex.to_openmm(),
                    self._complex.get_positions().to_openmm(),
                    f,
                )

    def __create_complex(self) -> None:
        self._complex = Topology(self.receptor)

        self.__complex_ligand_id = self._complex.add_molecule(
            Molecule.from_rdkit(
                self.ligand, allow_undefined_stereo=True, hydrogens_are_explicit=True
            )
        )
        self.__complex_pos = [
            m.conformers[0].to_openmm().value_in_unit(unit.angstrom)
            for m in self._complex.molecules
        ]

    def __update_complex(self) -> None:
        positions = from_openmm(
            self.__simulation.context.getState(positions=True).getPositions()
        )
        self._complex.set_positions(positions)
        self.__complex_pos = [
            m.conformers[0].to_openmm().value_in_unit(unit.angstrom)
            for m in self._complex.molecules
        ]

    def __update_positions(self) -> None:
        # Get ligand positions
        __ligand_pos = (
            Molecule.from_rdkit(
                self.ligand, allow_undefined_stereo=True, hydrogens_are_explicit=True
            )
            .to_topology()
            .get_positions()
        )
        # print(np.shape(__ligand_pos), np.shape(self.__complex_pos))
        self.__complex_pos[self.__complex_ligand_id] = np.array(
            __ligand_pos.to_openmm().value_in_unit(unit.angstrom)
        )
        pos = np.concat(self.__complex_pos)
        self.__simulation.context.setPositions(unit.Quantity(pos, unit.angstrom))
        self._complex.set_positions(from_openmm(unit.Quantity(pos, unit.angstrom)))

    def _score(self, *args, **kwargs) -> float:
        """Calculates and returns the potential energy as the score.

        Parameters
        ----------
        *args :

        **kwargs :


        Returns
        -------
        float
            A floating-point number representing the potential energy
            associated with the system.

        """
        return self._get_potential_energy()

    def _get_potential_energy(self) -> float:
        """ """
        self.__update_positions()
        state = self.__simulation.context.getState(energy=True)
        energy = state.getPotentialEnergy() / unit.kilocalories_per_mole
        if np.isnan(energy):
            energy = float("inf")
        return energy

    def __calc_indices(self) -> None:
        ligand_atoms = set(self._complex.molecule(self.__complex_ligand_id).atoms)
        for a in self._complex.atoms:
            if a in ligand_atoms:
                self.__ligand_indices.add(self._complex.atom_index(a))
            else:
                self.__protein_indices.add(self._complex.atom_index(a))

    def _prune_forces(self) -> None:
        """Prunes and updates the forces within the system object by identifying, removing,
        modifying, or adding forces. This process adjusts how certain force interactions
        are recognized or applied based on predefined criteria and indices of system components.
        It also categorizes specific force groups.

        Parameters
        ----------
        self :
            Instance of the class that contains system and force-related attributes
            necessary for pruning forces interactions.

        Returns
        -------
        type
            None

        """
        self.__calc_indices()
        forces = self.__system.getForces()

        intra_force_group = 1
        inter_force_group = 2

        force_ix_to_remove = []
        forces_to_add = []
        forces_to_add_exclusion = []

        for i, force in enumerate(forces):
            # Bond force: 713
            # Angle force: 3447
            if isinstance(
                force, (HarmonicBondForce, HarmonicAngleForce, CMMotionRemover)
            ):
                force_ix_to_remove.append(i)
                continue

            # Old force: 20100, new force: 47
            if isinstance(force, PeriodicTorsionForce):
                new_periodic_force = PeriodicTorsionForce()
                new_periodic_force.setForceGroup(intra_force_group)
                new_periodic_force.setUsesPeriodicBoundaryConditions(
                    force.usesPeriodicBoundaryConditions()
                )
                for torsion_i in range(force.getNumTorsions()):
                    p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(
                        torsion_i
                    )
                    if len(self.__ligand_indices & {p1, p2, p3, p4}) > 0:
                        new_periodic_force.addTorsion(
                            p1, p2, p3, p4, periodicity, phase, k
                        )
                force_ix_to_remove.append(i)
                forces_to_add.append(new_periodic_force)
                continue

            # Nonbonded Old force: -10000, new force: -100
            # CustomGB Old force: -75410.779, new force: -75410.777
            if isinstance(force, NonbondedForce) or isinstance(force, CustomGBForce):
                force.setForceGroup(inter_force_group)
                # Add all combinations of protein particles
                forces_to_add_exclusion.append(force)

        exclusion_pairs = itertools.combinations(self.__protein_indices, 2)
        for force in forces_to_add_exclusion:
            for p1, p2 in exclusion_pairs:
                if isinstance(force, NonbondedForce):
                    force.addException(p1, p2, 0, 0, 0, replace=True)
                elif isinstance(force, CustomGBForce):
                    force.addExclusion(p1, p2)

        for force_i in sorted(force_ix_to_remove, reverse=True):
            self.__system.removeForce(force_i)

        for force in forces_to_add:
            self.__system.addForce(force)
