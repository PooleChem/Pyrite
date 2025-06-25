from enum import IntEnum
from collections import namedtuple


class AtomType(IntEnum):
    """
    Enumeration describing the various atom types.

    The following types exist:

    * Unknown
    * Hydrogen
    * PolarHydrogen
    * AliphaticCarbonHydrophobe
    * AliphaticCarbonNonHydrophobe
    * AromaticCarbonHydrophobe
    * AromaticCarbonNonHydrophobe
    * Nitrogen
    * NitrogenDonor
    * NitrogenDonorAcceptor
    * NitrogenAcceptor
    * Oxygen
    * OxygenDonor
    * OxygenDonorAcceptor
    * OxygenAcceptor
    * Sulfur
    * SulfurAcceptor
    * Phosphorus
    * Fluorine
    * Chlorine
    * Bromine
    * Iodine
    * Magnesium
    * Manganese
    * Zinc
    * Calcium
    * Iron
    * GenericMetal
    * Boron



    .. note::
        ``AtomType`` is derived from `gnina <https://github.com/gnina/gnina>`_.
    """

    # pylint: disable=invalid-name

    Unknown = -1
    Hydrogen = 0
    PolarHydrogen = 1
    AliphaticCarbonHydrophobe = 2
    AliphaticCarbonNonHydrophobe = 3
    AromaticCarbonHydrophobe = 4
    AromaticCarbonNonHydrophobe = 5
    Nitrogen = 6
    NitrogenDonor = 7
    NitrogenDonorAcceptor = 8
    NitrogenAcceptor = 9
    Oxygen = 10
    OxygenDonor = 11
    OxygenDonorAcceptor = 12
    OxygenAcceptor = 13
    Sulfur = 14
    SulfurAcceptor = 15
    Phosphorus = 16
    Fluorine = 17
    Chlorine = 18
    Bromine = 19
    Iodine = 20
    Magnesium = 21
    Manganese = 22
    Zinc = 23
    Calcium = 24
    Iron = 25
    GenericMetal = 26
    Boron = 27

    def adjust(self, hbonded: bool, heterobonded: bool):
        """

        Parameters
        ----------
        hbonded: bool :

        heterobonded: bool :


        Returns
        -------

        """
        # Adjust aro/ali etc.
        if (
            self == AtomType.AliphaticCarbonHydrophobe
            or self == AtomType.AliphaticCarbonNonHydrophobe
        ):
            return (
                AtomType.AliphaticCarbonNonHydrophobe
                if heterobonded
                else AtomType.AliphaticCarbonHydrophobe
            )
        elif (
            self == AtomType.AromaticCarbonHydrophobe
            or self == AtomType.AromaticCarbonNonHydrophobe
        ):
            return (
                AtomType.AromaticCarbonNonHydrophobe
                if heterobonded
                else AtomType.AromaticCarbonHydrophobe
            )

        elif self == AtomType.NitrogenDonor or self == AtomType.Nitrogen:
            return AtomType.NitrogenDonor if hbonded else AtomType.Nitrogen
        elif (
            self == AtomType.NitrogenDonorAcceptor or self == AtomType.NitrogenAcceptor
        ):
            return (
                AtomType.NitrogenDonorAcceptor if hbonded else AtomType.NitrogenAcceptor
            )

        elif self == AtomType.OxygenDonor or self == AtomType.Oxygen:
            return AtomType.OxygenDonor if hbonded else AtomType.Oxygen
        elif self == AtomType.OxygenDonorAcceptor or self == AtomType.OxygenAcceptor:
            return AtomType.OxygenDonorAcceptor if hbonded else AtomType.OxygenAcceptor

        return self


Atom = namedtuple(
    "Atom",
    [
        "type",
        "smina_name",
        "ad_name",
        "anum",
        "ad_radius",
        "ad_depth",
        "ad_solvation",
        "ad_volume",
        "covalent_radius",
        "xs_radius",
        "xs_hydrophobe",
        "xs_donor",
        "xs_acceptor",
        "ad_heteroatom",
    ],
)


vina_atom_consts = {
    AtomType.Hydrogen: Atom(
        AtomType.Hydrogen,
        "Hydrogen",
        "H",
        1,
        1.000000,
        0.020000,
        0.000510,
        0.000000,
        0.370000,
        0.370000,
        False,
        False,
        False,
        False,
    ),
    AtomType.PolarHydrogen: Atom(
        AtomType.PolarHydrogen,
        "PolarHydrogen",
        "HD",
        1,
        1.000000,
        0.020000,
        0.000510,
        0.000000,
        0.370000,
        0.370000,
        False,
        False,
        False,
        False,
    ),
    AtomType.AliphaticCarbonHydrophobe: Atom(
        AtomType.AliphaticCarbonHydrophobe,
        "AliphaticCarbonHydrophobe",
        "C",
        6,
        2.000000,
        0.150000,
        -0.001430,
        33.510300,
        0.770000,
        1.900000,
        True,
        False,
        False,
        False,
    ),
    AtomType.AliphaticCarbonNonHydrophobe: Atom(
        AtomType.AliphaticCarbonNonHydrophobe,
        "AliphaticCarbonNonHydrophobe",
        "C",
        6,
        2.000000,
        0.150000,
        -0.001430,
        33.510300,
        0.770000,
        1.900000,
        False,
        False,
        False,
        False,
    ),
    AtomType.AromaticCarbonHydrophobe: Atom(
        AtomType.AromaticCarbonHydrophobe,
        "AromaticCarbonHydrophobe",
        "A",
        6,
        2.000000,
        0.150000,
        -0.000520,
        33.510300,
        0.770000,
        1.900000,
        True,
        False,
        False,
        False,
    ),
    AtomType.AromaticCarbonNonHydrophobe: Atom(
        AtomType.AromaticCarbonNonHydrophobe,
        "AromaticCarbonNonHydrophobe",
        "A",
        6,
        2.000000,
        0.150000,
        -0.000520,
        33.510300,
        0.770000,
        1.900000,
        False,
        False,
        False,
        False,
    ),
    AtomType.Nitrogen: Atom(
        AtomType.Nitrogen,
        "Nitrogen",
        "N",
        7,
        1.750000,
        0.160000,
        -0.001620,
        22.449300,
        0.750000,
        1.800000,
        False,
        False,
        False,
        True,
    ),
    AtomType.NitrogenDonor: Atom(
        AtomType.NitrogenDonor,
        "NitrogenDonor",
        "N",
        7,
        1.750000,
        0.160000,
        -0.001620,
        22.449300,
        0.750000,
        1.800000,
        False,
        True,
        False,
        True,
    ),
    AtomType.NitrogenDonorAcceptor: Atom(
        AtomType.NitrogenDonorAcceptor,
        "NitrogenDonorAcceptor",
        "NA",
        7,
        1.750000,
        0.160000,
        -0.001620,
        22.449300,
        0.750000,
        1.800000,
        False,
        True,
        True,
        True,
    ),
    AtomType.NitrogenAcceptor: Atom(
        AtomType.NitrogenAcceptor,
        "NitrogenAcceptor",
        "NA",
        7,
        1.750000,
        0.160000,
        -0.001620,
        22.449300,
        0.750000,
        1.800000,
        False,
        False,
        True,
        True,
    ),
    AtomType.Oxygen: Atom(
        AtomType.Oxygen,
        "Oxygen",
        "O",
        8,
        1.600000,
        0.200000,
        -0.002510,
        17.157300,
        0.730000,
        1.700000,
        False,
        False,
        False,
        True,
    ),
    AtomType.OxygenDonor: Atom(
        AtomType.OxygenDonor,
        "OxygenDonor",
        "O",
        8,
        1.600000,
        0.200000,
        -0.002510,
        17.157300,
        0.730000,
        1.700000,
        False,
        True,
        False,
        True,
    ),
    AtomType.OxygenDonorAcceptor: Atom(
        AtomType.OxygenDonorAcceptor,
        "OxygenDonorAcceptor",
        "OA",
        8,
        1.600000,
        0.200000,
        -0.002510,
        17.157300,
        0.730000,
        1.700000,
        False,
        True,
        True,
        True,
    ),
    AtomType.OxygenAcceptor: Atom(
        AtomType.OxygenAcceptor,
        "OxygenAcceptor",
        "OA",
        8,
        1.600000,
        0.200000,
        -0.002510,
        17.157300,
        0.730000,
        1.700000,
        False,
        False,
        True,
        True,
    ),
    AtomType.Sulfur: Atom(
        AtomType.Sulfur,
        "Sulfur",
        "S",
        16,
        2.000000,
        0.200000,
        -0.002140,
        33.510300,
        1.020000,
        2.000000,
        False,
        False,
        False,
        True,
    ),
    AtomType.SulfurAcceptor: Atom(
        AtomType.SulfurAcceptor,
        "SulfurAcceptor",
        "SA",
        16,
        2.000000,
        0.200000,
        -0.002140,
        33.510300,
        1.020000,
        2.000000,
        False,
        False,
        False,
        True,
    ),
    AtomType.Phosphorus: Atom(
        AtomType.Phosphorus,
        "Phosphorus",
        "P",
        15,
        2.100000,
        0.200000,
        -0.001100,
        38.792400,
        1.060000,
        2.100000,
        False,
        False,
        False,
        True,
    ),
    AtomType.Fluorine: Atom(
        AtomType.Fluorine,
        "Fluorine",
        "F",
        9,
        1.545000,
        0.080000,
        -0.001100,
        15.448000,
        0.710000,
        1.500000,
        True,
        False,
        False,
        True,
    ),
    AtomType.Chlorine: Atom(
        AtomType.Chlorine,
        "Chlorine",
        "Cl",
        17,
        2.045000,
        0.276000,
        -0.001100,
        35.823500,
        0.990000,
        1.800000,
        True,
        False,
        False,
        True,
    ),
    AtomType.Bromine: Atom(
        AtomType.Bromine,
        "Bromine",
        "Br",
        35,
        2.165000,
        0.389000,
        -0.001100,
        42.566100,
        1.140000,
        2.000000,
        True,
        False,
        False,
        True,
    ),
    AtomType.Iodine: Atom(
        AtomType.Iodine,
        "Iodine",
        "I",
        53,
        2.360000,
        0.550000,
        -0.001100,
        55.058500,
        1.330000,
        2.200000,
        True,
        False,
        False,
        True,
    ),
    AtomType.Magnesium: Atom(
        AtomType.Magnesium,
        "Magnesium",
        "Mg",
        12,
        0.650000,
        0.875000,
        -0.001100,
        1.560000,
        1.300000,
        1.200000,
        False,
        True,
        False,
        True,
    ),
    AtomType.Manganese: Atom(
        AtomType.Manganese,
        "Manganese",
        "Mn",
        25,
        0.650000,
        0.875000,
        -0.001100,
        2.140000,
        1.390000,
        1.200000,
        False,
        True,
        False,
        True,
    ),
    AtomType.Zinc: Atom(
        AtomType.Zinc,
        "Zinc",
        "Zn",
        30,
        0.740000,
        0.550000,
        -0.001100,
        1.700000,
        1.310000,
        1.200000,
        False,
        True,
        False,
        True,
    ),
    AtomType.Calcium: Atom(
        AtomType.Calcium,
        "Calcium",
        "Ca",
        20,
        0.990000,
        0.550000,
        -0.001100,
        2.770000,
        1.740000,
        1.200000,
        False,
        True,
        False,
        True,
    ),
    AtomType.Iron: Atom(
        AtomType.Iron,
        "Iron",
        "Fe",
        26,
        0.650000,
        0.010000,
        -0.001100,
        1.840000,
        1.250000,
        1.200000,
        False,
        True,
        False,
        True,
    ),
    AtomType.GenericMetal: Atom(
        AtomType.GenericMetal,
        "GenericMetal",
        "M",
        0,
        1.200000,
        0.000000,
        -0.001100,
        22.449300,
        1.750000,
        1.200000,
        False,
        True,
        False,
        True,
    ),
    AtomType.Boron: Atom(
        AtomType.Boron,
        "Boron",
        "B",
        5,
        2.04,
        0.180000,
        -0.0011,
        12.052,
        0.90,
        1.920000,
        True,
        False,
        False,
        False,
    ),
}
