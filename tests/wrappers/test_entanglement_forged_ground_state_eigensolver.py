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
# pylint: disable=wrong-import-position
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
from qiskit_nature.algorithms.ground_state_solvers import (
    GroundStateEigensolver,
    NumPyMinimumEigensolverFactory,
)
from qiskit_nature.transformers.second_quantization.electronic.active_space_transformer import (
    ActiveSpaceTransformer,
)
from qiskit_nature import settings

settings.dict_aux_operators = True

from entanglement_forging import reduce_bitstrings
from entanglement_forging import (
    EntanglementForgedConfig,
    EntanglementForgedDriver,
    EntanglementForgedGroundStateSolver,
)


class TestEntanglementForgedGroundStateEigensolver(unittest.TestCase):
    """EntanglementForgedGroundStateEigensolver tests."""

    def setUp(self):
        np.random.seed(42)
        self.backend = BasicAer.get_backend("statevector_simulator")

    def test_forged_vqe_for_hydrogen(self):
        """Test of applying Entanglement Forged VQE to to compute the energy of a H2 molecule."""
        # setup problem
        molecule = Molecule(
            geometry=[("H", [0.0, 0.0, 0.0]), ("H", [0.0, 0.0, 0.735])],
            charge=0,
            multiplicity=1,
        )
        driver = PySCFDriver.from_molecule(molecule)
        problem = ElectronicStructureProblem(driver)
        problem.second_q_ops()

        # solution
        bitstrings = [[1, 0], [0, 1]]
        ansatz = TwoLocal(2, [], "cry", [[0, 1], [1, 0]], reps=1)

        config = EntanglementForgedConfig(
            backend=self.backend, maxiter=0, initial_params=[0, 0.5 * np.pi]
        )

        converter = QubitConverter(JordanWignerMapper())

        forged_ground_state_solver = EntanglementForgedGroundStateSolver(
            converter, ansatz, bitstrings, config
        )

        forged_result = forged_ground_state_solver.solve(problem)

        self.assertAlmostEqual(forged_result.ground_state_energy, -1.1219365445030705)

    def test_forged_vqe_for_water(self):  # pylint: disable=too-many-locals
        """Test of applying Entanglement Forged VQE to to compute the energy of a H20 molecule."""
        # setup problem
        radius_1 = 0.958  # position for the first H atom
        radius_2 = 0.958  # position for the second H atom
        thetas_in_deg = 104.478  # bond angles.

        h1_x = radius_1
        h2_x = radius_2 * np.cos(np.pi / 180 * thetas_in_deg)
        h2_y = radius_2 * np.sin(np.pi / 180 * thetas_in_deg)

        molecule = Molecule(
            geometry=[
                ("O", [0.0, 0.0, 0.0]),
                ("H", [h1_x, 0.0, 0.0]),
                ("H", [h2_x, h2_y, 0.0]),
            ],
            charge=0,
            multiplicity=1,
        )
        driver = PySCFDriver.from_molecule(molecule, basis="sto6g")
        problem = ElectronicStructureProblem(driver)
        problem.second_q_ops()

        # solution
        orbitals_to_reduce = [0, 3]
        bitstrings = [
            [1, 1, 1, 1, 1, 0, 0],
            [1, 0, 1, 1, 1, 0, 1],
            [1, 0, 1, 1, 1, 1, 0],
        ]
        reduced_bitstrings = reduce_bitstrings(bitstrings, orbitals_to_reduce)

        theta = Parameter("θ")
        theta_1, theta_2, theta_3, theta_4 = (
            Parameter("θ1"),
            Parameter("θ2"),
            Parameter("θ3"),
            Parameter("θ4"),
        )

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
            backend=self.backend,
            maxiter=0,
            spsa_c0=20 * np.pi,
            initial_params=[0, 0, 0, 0],
        )

        converter = QubitConverter(JordanWignerMapper())

        solver = EntanglementForgedGroundStateSolver(
            converter,
            ansatz,
            reduced_bitstrings,
            config,
            orbitals_to_reduce=orbitals_to_reduce,
        )
        forged_result = solver.solve(problem)
        self.assertAlmostEqual(forged_result.ground_state_energy, -75.68366174497027)

    def test_ef_driver(self):
        """Test for entanglement forging driver."""
        hcore = np.array([[-1.12421758, -0.9652574], [-0.9652574, -1.12421758]])
        mo_coeff = np.array([[0.54830202, 1.21832731], [0.54830202, -1.21832731]])
        eri = np.array(
            [
                [
                    [[0.77460594, 0.44744572], [0.44744572, 0.57187698]],
                    [[0.44744572, 0.3009177], [0.3009177, 0.44744572]],
                ],
                [
                    [[0.44744572, 0.3009177], [0.3009177, 0.44744572]],
                    [[0.57187698, 0.44744572], [0.44744572, 0.77460594]],
                ],
            ]
        )

        driver = EntanglementForgedDriver(
            hcore=hcore,
            mo_coeff=mo_coeff,
            eri=eri,
            num_alpha=1,
            num_beta=1,
            nuclear_repulsion_energy=0.7199689944489797,
        )
        problem = ElectronicStructureProblem(driver)
        problem.second_q_ops()

        bitstrings = [[1, 0], [0, 1]]
        ansatz = TwoLocal(2, [], "cry", [[0, 1], [1, 0]], reps=1)

        config = EntanglementForgedConfig(
            backend=self.backend, maxiter=0, initial_params=[0, 0.5 * np.pi]
        )
        converter = QubitConverter(JordanWignerMapper())
        forged_ground_state_solver = EntanglementForgedGroundStateSolver(
            converter, ansatz, bitstrings, config
        )
        forged_result = forged_ground_state_solver.solve(problem)
        self.assertAlmostEqual(forged_result.ground_state_energy, -1.1219365445030705)

    def test_ground_state_eigensolver_with_ef_driver(self):
        """Tests standard qiskit nature solver."""
        hcore = np.array([[-1.12421758, -0.9652574], [-0.9652574, -1.12421758]])
        mo_coeff = np.array([[0.54830202, 1.21832731], [0.54830202, -1.21832731]])
        eri = np.array(
            [
                [
                    [[0.77460594, 0.44744572], [0.44744572, 0.57187698]],
                    [[0.44744572, 0.3009177], [0.3009177, 0.44744572]],
                ],
                [
                    [[0.44744572, 0.3009177], [0.3009177, 0.44744572]],
                    [[0.57187698, 0.44744572], [0.44744572, 0.77460594]],
                ],
            ]
        )

        repulsion_energy = 0.7199689944489797
        driver = EntanglementForgedDriver(
            hcore=hcore,
            mo_coeff=mo_coeff,
            eri=eri,
            num_alpha=1,
            num_beta=1,
            nuclear_repulsion_energy=repulsion_energy,
        )
        problem = ElectronicStructureProblem(driver)
        problem.second_q_ops()

        converter = QubitConverter(JordanWignerMapper())
        solver = GroundStateEigensolver(
            converter,
            NumPyMinimumEigensolverFactory(use_default_filter_criterion=False),
        )
        result = solver.solve(problem)
        self.assertAlmostEqual(
            -1.137306026563, np.real(result.eigenenergies[0]) + repulsion_energy
        )

    def test_hydrogen_duplicate_bitstrings(self):
        """Test of applying Entanglement Forged VQE to to compute the energy of a H2 molecule."""
        # setup problem
        molecule = Molecule(
            geometry=[("H", [0.0, 0.0, 0.0]), ("H", [0.0, 0.0, 0.735])],
            charge=0,
            multiplicity=1,
        )
        driver = PySCFDriver.from_molecule(molecule)
        problem = ElectronicStructureProblem(driver)
        problem.second_q_ops()

        # solution
        bitstrings = [[1, 0], [0, 1]]
        ansatz = TwoLocal(2, [], "cry", [[0, 1], [1, 0]], reps=1)

        config = EntanglementForgedConfig(
            backend=self.backend, maxiter=0, initial_params=[0, 0.5 * np.pi]
        )

        converter = QubitConverter(JordanWignerMapper())

        forged_ground_state_solver = EntanglementForgedGroundStateSolver(
            converter, ansatz, bitstrings, config, bitstrings_v=bitstrings
        )

        forged_result = forged_ground_state_solver.solve(problem)

        self.assertAlmostEqual(forged_result.ground_state_energy, -1.1219365445030705)

    def test_asymmetric_bitstrings(self):
        # Setting up the entanglement forging problem in the 4 orbital active space:

        # Construct the One and two body integrals
        one_body_integrals_alpha = np.array([
            [-1.12861898e00, -1.15203518e-05, 3.63968090e-03, 2.32157320e-06],
            [-1.15203518e-05, -1.10523006e00, -8.22602690e-05, 2.51250974e-02],
            [3.63968090e-03, -8.22602690e-05, -8.41006046e-01, 2.16626269e-05],
            [2.32157321e-06, 2.51250974e-02, 2.16626269e-05, -8.19960466e-01],
        ])

        two_body_integrals_alpha_alpha = np.array([
            [
                [
                    [3.68276791e-01, 1.19118374e-04, 2.50664958e-02, 9.85918985e-05],
                    [1.19118374e-04, 2.66339792e-01, 1.16646555e-04, -2.22635970e-02],
                    [2.50664958e-02, 1.16646555e-04, 2.59956713e-01, -9.92839111e-05],
                    [9.85918985e-05, -2.22635970e-02, -9.92839111e-05, 3.59795290e-01],
                ],
                [
                    [1.19118374e-04, 2.39880285e-02, 7.11940570e-05, -2.78018790e-03],
                    [2.39880285e-02, -1.07587175e-04, 6.95089611e-03, -7.41706108e-05],
                    [7.11940570e-05, 6.95089611e-03, -1.63888338e-04, -2.23331398e-02],
                    [-2.78018790e-03, -7.41706108e-05, -2.23331398e-02, 1.11039153e-04],
                ],
                [
                    [2.50664958e-02, 7.11940570e-05, 3.72212438e-02, -5.12218979e-05],
                    [7.11940570e-05, -1.08776665e-02, -9.22315145e-05, -3.85171770e-02],
                    [3.72212438e-02, -9.22315145e-05, -1.12131432e-02, -8.84397562e-05],
                    [-5.12218979e-05, -3.85171770e-02, -8.84397562e-05, 1.67284550e-02],
                ],
                [
                    [9.85918985e-05, -2.78018790e-03, -5.12218979e-05, 1.00666612e-01],
                    [-2.78018790e-03, -8.75941750e-05, 4.11933131e-03, 6.92867070e-05],
                    [-5.12218979e-05, 4.11933131e-03, -1.15058864e-04, 1.00004060e-05],
                    [1.00666612e-01, 6.92867070e-05, 1.00004060e-05, -1.80321694e-05],
                ],
            ],
            [
                [
                    [1.19118374e-04, 2.39880285e-02, 7.11940570e-05, -2.78018790e-03],
                    [2.39880285e-02, -1.07587175e-04, 6.95089611e-03, -7.41706108e-05],
                    [7.11940570e-05, 6.95089611e-03, -1.63888338e-04, -2.23331398e-02],
                    [-2.78018790e-03, -7.41706108e-05, -2.23331398e-02, 1.11039153e-04],
                ],
                [
                    [2.66339792e-01, -1.07587175e-04, -1.08776665e-02, -8.75941750e-05],
                    [-1.07587175e-04, 3.58305904e-01, -7.99295362e-05, 1.66219712e-02],
                    [-1.08776665e-02, -7.99295362e-05, 3.52657056e-01, 7.20630473e-05],
                    [-8.75941750e-05, 1.66219712e-02, 7.20630473e-05, 2.73778885e-01],
                ],
                [
                    [1.16646555e-04, 6.95089611e-03, -9.22315145e-05, 4.11933131e-03],
                    [6.95089611e-03, -7.99295362e-05, 9.14432014e-02, 1.84490574e-05],
                    [-9.22315145e-05, 9.14432014e-02, -1.28874923e-04, -6.19805823e-03],
                    [4.11933131e-03, 1.84490574e-05, -6.19805823e-03, 9.96369794e-05],
                ],
                [
                    [-2.22635970e-02, -7.41706108e-05, -3.85171770e-02, 6.92867070e-05],
                    [-7.41706108e-05, 1.66219712e-02, 1.84490574e-05, 4.42135787e-02],
                    [-3.85171770e-02, 1.84490574e-05, 1.50748142e-02, 9.37267247e-05],
                    [6.92867070e-05, 4.42135787e-02, 9.37267247e-05, -1.48329006e-02],
                ],
            ],
            [
                [
                    [2.50664958e-02, 7.11940570e-05, 3.72212438e-02, -5.12218979e-05],
                    [7.11940570e-05, -1.08776665e-02, -9.22315145e-05, -3.85171770e-02],
                    [3.72212438e-02, -9.22315145e-05, -1.12131432e-02, -8.84397562e-05],
                    [-5.12218979e-05, -3.85171770e-02, -8.84397562e-05, 1.67284550e-02],
                ],
                [
                    [1.16646555e-04, 6.95089611e-03, -9.22315145e-05, 4.11933131e-03],
                    [6.95089611e-03, -7.99295362e-05, 9.14432014e-02, 1.84490574e-05],
                    [-9.22315145e-05, 9.14432014e-02, -1.28874923e-04, -6.19805823e-03],
                    [4.11933131e-03, 1.84490574e-05, -6.19805823e-03, 9.96369794e-05],
                ],
                [
                    [2.59956713e-01, -1.63888338e-04, -1.12131432e-02, -1.15058864e-04],
                    [-1.63888338e-04, 3.52657056e-01, -1.28874923e-04, 1.50748142e-02],
                    [-1.12131432e-02, -1.28874923e-04, 3.54797171e-01, 1.31085482e-04],
                    [-1.15058864e-04, 1.50748142e-02, 1.31085482e-04, 2.66141901e-01],
                ],
                [
                    [-9.92839111e-05, -2.23331398e-02, -8.84397562e-05, 1.00004060e-05],
                    [-2.23331398e-02, 7.20630473e-05, -6.19805823e-03, 9.37267247e-05],
                    [-8.84397562e-05, -6.19805823e-03, 1.31085482e-04, 2.33628274e-02],
                    [1.00004060e-05, 9.37267247e-05, 2.33628274e-02, -9.31478640e-05],
                ],
            ],
            [
                [
                    [9.85918985e-05, -2.78018790e-03, -5.12218979e-05, 1.00666612e-01],
                    [-2.78018790e-03, -8.75941750e-05, 4.11933131e-03, 6.92867070e-05],
                    [-5.12218979e-05, 4.11933131e-03, -1.15058864e-04, 1.00004060e-05],
                    [1.00666612e-01, 6.92867070e-05, 1.00004060e-05, -1.80321694e-05],
                ],
                [
                    [-2.22635970e-02, -7.41706108e-05, -3.85171770e-02, 6.92867070e-05],
                    [-7.41706108e-05, 1.66219712e-02, 1.84490574e-05, 4.42135787e-02],
                    [-3.85171770e-02, 1.84490574e-05, 1.50748142e-02, 9.37267247e-05],
                    [6.92867070e-05, 4.42135787e-02, 9.37267247e-05, -1.48329006e-02],
                ],
                [
                    [-9.92839111e-05, -2.23331398e-02, -8.84397562e-05, 1.00004060e-05],
                    [-2.23331398e-02, 7.20630473e-05, -6.19805823e-03, 9.37267247e-05],
                    [-8.84397562e-05, -6.19805823e-03, 1.31085482e-04, 2.33628274e-02],
                    [1.00004060e-05, 9.37267247e-05, 2.33628274e-02, -9.31478640e-05],
                ],
                [
                    [3.59795290e-01, 1.11039153e-04, 1.67284550e-02, -1.80321694e-05],
                    [1.11039153e-04, 2.73778885e-01, 9.96369794e-05, -1.48329006e-02],
                    [1.67284550e-02, 9.96369794e-05, 2.66141901e-01, -9.31478640e-05],
                    [-1.80321694e-05, -1.48329006e-02, -9.31478640e-05, 3.60696220e-01],
                ],
            ],
        ])

        nuclear_repulsion_energy = -264.7518219120776

        driver = EntanglementForgedDriver(
            hcore=one_body_integrals_alpha,
            mo_coeff=np.eye(4, 4),
            eri=two_body_integrals_alpha_alpha,
            num_alpha=2,
            num_beta=2,
            nuclear_repulsion_energy=nuclear_repulsion_energy,
        )
        problem = ElectronicStructureProblem(driver)
        problem.second_q_ops()

        converter = QubitConverter(JordanWignerMapper())

        orbitals_to_reduce = []

        # Feature request: ability to input asymmetric bitstrings with the same number of qubits.

        bitstrings_u = [[1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 1, 0]]
        bitstrings_v = [[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1]]

        n_theta = 1
        theta = Parameter("θ")
        mock_gate = QuantumCircuit(1, name="mock gate")
        mock_gate.rz(theta, 0)

        theta_vec = [Parameter("θ%d" % i) for i in range(n_theta)]
        nqubit = len(bitstrings_u[0])
        ansatz = QuantumCircuit(nqubit)
        ansatz.append(mock_gate.to_gate({theta: theta_vec[0]}), [0])

        config = EntanglementForgedConfig(
            backend=self.backend,
            maxiter=0,
            initial_params=[0.0] * n_theta,
            optimizer_name="COBYLA",
        )

        calc = EntanglementForgedGroundStateSolver(
            converter,
            ansatz,
            bitstrings_u=bitstrings_u,
            bitstrings_v=bitstrings_v,
            config=config,
            orbitals_to_reduce=orbitals_to_reduce,
        )
        res = calc.solve(problem)
        self.assertAlmostEqual(res.ground_state_energy, -267.48257702668974)

        # Test special case where bitstring lists contain duplicates
        bitstrings_u = [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]]
        bitstrings_v = [[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 0, 0]]

        calc = EntanglementForgedGroundStateSolver(
            converter,
            ansatz,
            bitstrings_u=bitstrings_u,
            bitstrings_v=bitstrings_v,
            config=config,
            orbitals_to_reduce=orbitals_to_reduce,
        )
        res = calc.solve(problem)
        self.assertAlmostEqual(res.ground_state_energy, -267.4819703739051)
