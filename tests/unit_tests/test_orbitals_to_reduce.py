"""Unit tests for OrbitalsToReduce object."""
import unittest

import numpy as np
from qiskit_nature.drivers import PySCFDriver, Molecule
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem

from entanglement_forging import OrbitalsToReduce


class TestEntanglementForgedGroundStateEigensolver(unittest.TestCase):
    """ EntanglementForgedGroundStateEigensolver tests. """

    def test_orbitals_to_reduce_water_all(self):
        """ Test for when we have both occupied and virtual orbitals.  """
        # setup problem
        radius_1 = 0.958  # position for the first H atom
        radius_2 = 0.958  # position for the second H atom
        thetas_in_deg = 104.478  # bond angles.

        hydrogen_1_x_coord = radius_1
        hydrogen_2_x_coord = radius_2*np.cos(np.pi/180 * thetas_in_deg)
        hydrogen_2_y_coord = radius_2*np.sin(np.pi/180 * thetas_in_deg)

        molecule = Molecule(geometry=[['O', [0., 0., 0.]],
                                    ['H', [hydrogen_1_x_coord, 0., 0.]],
                                    ['H', [hydrogen_2_x_coord, hydrogen_2_y_coord, 0.0]]],
                                    charge=0, multiplicity=1)
        driver = PySCFDriver(molecule = molecule, basis='sto6g')
        problem = ElectronicStructureProblem(driver)
        qmolecule = problem.driver.run()

        all_orbitals_to_reduce = [0,1,2,3,4,5,6,7,8]

        # solution
        orbitals_to_reduce = OrbitalsToReduce(all_orbitals_to_reduce, qmolecule)
        self.assertAlmostEqual(orbitals_to_reduce.occupied(), [0, 1, 2, 3, 4])
        self.assertAlmostEqual(orbitals_to_reduce.virtual(), [5, 6, 7, 8])

    def test_orbitals_to_reduce_water_occupied(self):
        """ Test for when we have only occupied orbitals.  """
        # setup problem
        radius_1 = 0.958  # position for the first H atom
        radius_2 = 0.958  # position for the second H atom
        thetas_in_deg = 104.478  # bond angles.

        hydrogen_1_x_coord = radius_1
        hydrogen_2_x_coord = radius_2*np.cos(np.pi/180 * thetas_in_deg)
        hydrogen_2_y_coord = radius_2*np.sin(np.pi/180 * thetas_in_deg)

        molecule = Molecule(geometry=[['O', [0., 0., 0.]],
                                    ['H', [hydrogen_1_x_coord, 0., 0.]],
                                    ['H', [hydrogen_2_x_coord, hydrogen_2_y_coord, 0.0]]],
                                    charge=0, multiplicity=1)
        driver = PySCFDriver(molecule = molecule, basis='sto6g')
        problem = ElectronicStructureProblem(driver)
        qmolecule = problem.driver.run()

        all_orbitals_to_reduce = [0,2,4]

        # solution
        orbitals_to_reduce = OrbitalsToReduce(all_orbitals_to_reduce, qmolecule)
        self.assertAlmostEqual(orbitals_to_reduce.occupied(), [0, 2, 4])
        self.assertAlmostEqual(orbitals_to_reduce.virtual(), [])

    def test_orbitals_to_reduce_water_virtual(self):
        """ Test for when we have only virtual orbitals.  """
        # setup problem
        radius_1 = 0.958  # position for the first H atom
        radius_2 = 0.958  # position for the second H atom
        thetas_in_deg = 104.478  # bond angles.

        hydrogen_1_x_coord = radius_1
        hydrogen_2_x_coord = radius_2*np.cos(np.pi/180 * thetas_in_deg)
        hydrogen_2_y_coord = radius_2*np.sin(np.pi/180 * thetas_in_deg)

        molecule = Molecule(geometry=[['O', [0., 0., 0.]],
                                    ['H', [hydrogen_1_x_coord, 0., 0.]],
                                    ['H', [hydrogen_2_x_coord, hydrogen_2_y_coord, 0.0]]],
                                    charge=0, multiplicity=1)
        driver = PySCFDriver(molecule = molecule, basis='sto6g')
        problem = ElectronicStructureProblem(driver)
        qmolecule = problem.driver.run()

        all_orbitals_to_reduce = [6,7,9]

        # solution
        orbitals_to_reduce = OrbitalsToReduce(all_orbitals_to_reduce, qmolecule)
        self.assertAlmostEqual(orbitals_to_reduce.occupied(), [])
        self.assertAlmostEqual(orbitals_to_reduce.virtual(), [6,7,9])
