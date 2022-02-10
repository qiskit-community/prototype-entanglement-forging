# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Integration tests for EntanglementForgedVQE module."""
import unittest

import numpy as np
from qiskit import BasicAer
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.drivers import Molecule
from qiskit_nature.drivers.second_quantization import PySCFDriver
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.algorithms.ground_state_solvers import (GroundStateEigensolver,
                                                           NumPyMinimumEigensolverFactory)
from qiskit_nature import settings
settings.dict_aux_operators = True

from entanglement_forging import (EntanglementForgedConfig, EntanglementForgedDriver,
                                  EntanglementForgedGroundStateSolver)
from entanglement_forging import reduce_bitstrings

class TestEntanglementForgedGroundStateEigensolver(unittest.TestCase):
    """ EntanglementForgedGroundStateEigensolver tests. """

    def setUp(self):
        np.random.seed(42)
        self.backend = BasicAer.get_backend("statevector_simulator")

    def test_forged_vqe_for_hydrogen(self):
        """ Test of applying Entanglement Forged VQE to to compute the energy of a H2 molecule. """
        # setup problem
        molecule = Molecule(geometry=[('H', [0., 0., 0.]),
                                      ('H', [0., 0., 0.735])],
                            charge=0, multiplicity=1)
        driver = PySCFDriver.from_molecule(molecule)
        problem = ElectronicStructureProblem(driver)
        driver_result = problem.second_q_ops()

        # solution
        bitstrings = [[1, 0], [0, 1]]
        ansatz = TwoLocal(2, [], 'cry', [[0, 1], [1, 0]], reps=1)

        config = EntanglementForgedConfig(backend=self.backend,
                                          maxiter=0,
                                          initial_params=[0, 0.5 * np.pi])

        converter = QubitConverter(JordanWignerMapper())

        forged_ground_state_solver = EntanglementForgedGroundStateSolver(
            converter, ansatz, bitstrings, config)

        forged_result = forged_ground_state_solver.solve(problem)

        self.assertAlmostEqual(forged_result.ground_state_energy, -1.1219365445030705)

    def test_forged_vqe_for_water(self):  # pylint: disable=too-many-locals
        """ Test of applying Entanglement Forged VQE to to compute the energy of a H20 molecule. """
        # setup problem
        radius_1 = 0.958  # position for the first H atom
        radius_2 = 0.958  # position for the second H atom
        thetas_in_deg = 104.478  # bond angles.

        h1_x = radius_1
        h2_x = radius_2 * np.cos(np.pi / 180 * thetas_in_deg)
        h2_y = radius_2 * np.sin(np.pi / 180 * thetas_in_deg)

        molecule = Molecule(geometry=[('O', [0., 0., 0.]),
                                      ('H', [h1_x, 0., 0.]),
                                      ('H', [h2_x, h2_y, 0.0])], charge=0, multiplicity=1)
        driver = PySCFDriver.from_molecule(molecule, basis='sto6g')
        problem = ElectronicStructureProblem(driver)
        driver_result = problem.second_q_ops()

        # solution
        orbitals_to_reduce = [0, 3]
        bitstrings = [[1, 1, 1, 1, 1, 0, 0],
                      [1, 0, 1, 1, 1, 0, 1],
                      [1, 0, 1, 1, 1, 1, 0]]
        reduced_bitstrings = reduce_bitstrings(bitstrings, orbitals_to_reduce)

        theta = Parameter('θ')
        theta_1, theta_2, theta_3, theta_4 = Parameter('θ1'), Parameter('θ2'), \
                                             Parameter('θ3'), Parameter('θ4')

        hop_gate = QuantumCircuit(2, name="Hop gate")
        hop_gate.h(0)
        hop_gate.cx(1, 0)
        hop_gate.cx(0, 1)
        hop_gate.ry(-theta, 0)
        hop_gate.ry(-theta, 1)
        hop_gate.cx(0, 1)
        hop_gate.h(0)

        ansatz = QuantumCircuit(5)
        ansatz.append(hop_gate.to_gate({theta: theta_1}), [0, 1])
        ansatz.append(hop_gate.to_gate({theta: theta_2}), [3, 4])
        ansatz.append(hop_gate.to_gate({theta: 0}), [1, 4])
        ansatz.append(hop_gate.to_gate({theta: theta_3}), [0, 2])
        ansatz.append(hop_gate.to_gate({theta: theta_4}), [3, 4])

        config = EntanglementForgedConfig(
            backend=self.backend, maxiter=0, spsa_c0=20 * np.pi, initial_params=[0, 0, 0, 0])

        converter = QubitConverter(JordanWignerMapper())

        solver = EntanglementForgedGroundStateSolver(converter, ansatz, reduced_bitstrings, config,
                                                     orbitals_to_reduce)
        forged_result = solver.solve(problem)
        self.assertAlmostEqual(forged_result.ground_state_energy, -75.68366174497027)

    def test_ef_driver(self):
        """Test for entanglement forging driver."""
        hcore = np.array([
            [-1.12421758, -0.9652574],
            [-0.9652574, - 1.12421758]
        ])
        mo_coeff = np.array([
            [0.54830202, 1.21832731],
            [0.54830202, -1.21832731]
        ])
        eri = np.array([
            [[[0.77460594, 0.44744572], [0.44744572, 0.57187698]],
             [[0.44744572, 0.3009177], [0.3009177, 0.44744572]]],
            [[[0.44744572, 0.3009177], [0.3009177, 0.44744572]],
             [[0.57187698, 0.44744572], [0.44744572, 0.77460594]]]
        ])

        driver = EntanglementForgedDriver(hcore=hcore,
                                          mo_coeff=mo_coeff,
                                          eri=eri,
                                          num_alpha=1,
                                          num_beta=1,
                                          nuclear_repulsion_energy=0.7199689944489797)
        problem = ElectronicStructureProblem(driver)
        driver_result = problem.second_q_ops()

        bitstrings = [[1, 0], [0, 1]]
        ansatz = TwoLocal(2, [], 'cry', [[0, 1], [1, 0]], reps=1)

        config = EntanglementForgedConfig(backend=self.backend,
                                          maxiter=0,
                                          initial_params=[0, 0.5 * np.pi])
        converter = QubitConverter(JordanWignerMapper())
        forged_ground_state_solver = EntanglementForgedGroundStateSolver(
            converter, ansatz, bitstrings, config)
        forged_result = forged_ground_state_solver.solve(problem)
        self.assertAlmostEqual(forged_result.ground_state_energy, -1.1219365445030705)

    def test_aux_results(self):
        """Test for aux results data.
        NOTE: aux data was added only because of before this data was stored in file system,
              possible it is not needed at all and can be removed
        """
        # setup problem
        molecule = Molecule(geometry=[('H', [0., 0., 0.]),
                                      ('H', [0., 0., 0.735])],
                            charge=0, multiplicity=1)
        driver = PySCFDriver.from_molecule(molecule)
        problem = ElectronicStructureProblem(driver)
        driver_result = problem.second_q_ops()

        # solution
        bitstrings = [[1, 0], [0, 1]]
        ansatz = TwoLocal(2, [], 'cry', [[0, 1], [1, 0]], reps=1)

        config = EntanglementForgedConfig(backend=self.backend,
                                          maxiter=0,
                                          initial_params=[0, 0.5 * np.pi])
        converter = QubitConverter(JordanWignerMapper())
        forged_ground_state_solver = EntanglementForgedGroundStateSolver(
            converter, ansatz, bitstrings, config)
        forged_result = forged_ground_state_solver.solve(problem)

        self.assertEqual([name for name, _ in forged_result.auxiliary_results],
                         ['bootstrap', 'data', 'data_noextrapolation', 'optimal_params'])

    def test_ground_state_eigensolver_with_ef_driver(self):
        """Tests standard qiskit nature solver."""
        hcore = np.array([
            [-1.12421758, -0.9652574],
            [-0.9652574, - 1.12421758]
        ])
        mo_coeff = np.array([
            [0.54830202, 1.21832731],
            [0.54830202, -1.21832731]
        ])
        eri = np.array([
            [[[0.77460594, 0.44744572], [0.44744572, 0.57187698]],
             [[0.44744572, 0.3009177], [0.3009177, 0.44744572]]],
            [[[0.44744572, 0.3009177], [0.3009177, 0.44744572]],
             [[0.57187698, 0.44744572], [0.44744572, 0.77460594]]]
        ])

        repulsion_energy = 0.7199689944489797
        driver = EntanglementForgedDriver(hcore=hcore,
                                          mo_coeff=mo_coeff,
                                          eri=eri,
                                          num_alpha=1,
                                          num_beta=1,
                                          nuclear_repulsion_energy=repulsion_energy)
        problem = ElectronicStructureProblem(driver)
        driver_result = problem.second_q_ops()

        converter = QubitConverter(JordanWignerMapper())
        solver = GroundStateEigensolver(converter,
                                        NumPyMinimumEigensolverFactory(
                                            use_default_filter_criterion=False))
        result = solver.solve(problem)
        self.assertAlmostEqual(-1.137306026563, np.real(result.eigenenergies[0]) + repulsion_energy)
