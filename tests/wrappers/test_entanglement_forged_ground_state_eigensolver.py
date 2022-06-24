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
import os

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
        self.config = EntanglementForgedConfig(
            backend=self.backend,
            maxiter=0,
            initial_params=[0.0],
            optimizer_name="COBYLA",
        )

        self.mock_ts_ansatz = self.create_mock_ansatz(4)
        self.hcore_ts = np.load(
            os.path.join(os.path.dirname(__file__), "test_data", "TS_one_body.npy")
        )
        self.eri_ts = np.load(
            os.path.join(os.path.dirname(__file__), "test_data", "TS_two_body.npy")
        )
        self.energy_shift_ts = -264.7518219120776

        self.mock_o2_ansatz = self.create_mock_ansatz(8)
        self.hcore_o2 = np.load(
            os.path.join(os.path.dirname(__file__), "test_data", "O2_one_body.npy")
        )
        self.eri_o2 = np.load(
            os.path.join(os.path.dirname(__file__), "test_data", "O2_two_body.npy")
        )
        self.energy_shift_o2 = -99.83894101027317

        self.mock_cn_ansatz = self.create_mock_ansatz(8)
        self.hcore_cn = np.load(
            os.path.join(os.path.dirname(__file__), "test_data", "CN_one_body.npy")
        )
        self.eri_cn = np.load(
            os.path.join(os.path.dirname(__file__), "test_data", "CN_two_body.npy")
        )
        self.energy_shift_cn = -67.185615568892

        self.mock_ch3_ansatz = self.create_mock_ansatz(6)
        self.hcore_ch3 = np.load(
            os.path.join(os.path.dirname(__file__), "test_data", "CH3_one_body.npy")
        )
        self.eri_ch3 = np.load(
            os.path.join(os.path.dirname(__file__), "test_data", "CH3_two_body.npy")
        )
        self.energy_shift_ch3 = -31.90914780401554

    def create_mock_ansatz(self, num_qubits):
        n_theta = 1
        theta = Parameter("θ")
        mock_gate = QuantumCircuit(1, name="mock gate")
        mock_gate.rz(theta, 0)

        theta_vec = [Parameter("θ%d" % i) for i in range(1)]
        ansatz = QuantumCircuit(num_qubits)
        ansatz.append(mock_gate.to_gate({theta: theta_vec[0]}), [0])

        return ansatz

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

    def test_TS_1(self):
        driver = EntanglementForgedDriver(
            hcore=self.hcore_ts,
            mo_coeff=np.eye(4, 4),
            eri=self.eri_ts,
            num_alpha=2,
            num_beta=2,
            nuclear_repulsion_energy=self.energy_shift_ts,
        )
        problem = ElectronicStructureProblem(driver)
        problem.second_q_ops()

        converter = QubitConverter(JordanWignerMapper())

        bitstrings_u = [[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1]]
        bitstrings_v = [[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1]]

        calc = EntanglementForgedGroundStateSolver(
            converter,
            self.mock_ts_ansatz,
            bitstrings_u=bitstrings_u,
            bitstrings_v=bitstrings_v,
            config=self.config,
            orbitals_to_reduce=[],
        )
        res = calc.solve(problem)
        self.assertAlmostEqual(-267.5081390468769, res.ground_state_energy)

    def test_TS_2(self):
        driver = EntanglementForgedDriver(
            hcore=self.hcore_ts,
            mo_coeff=np.eye(4, 4),
            eri=self.eri_ts,
            num_alpha=2,
            num_beta=2,
            nuclear_repulsion_energy=self.energy_shift_ts,
        )
        problem = ElectronicStructureProblem(driver)
        problem.second_q_ops()

        converter = QubitConverter(JordanWignerMapper())

        bitstrings_u = [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
        ]
        bitstrings_v = [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
        ]

        calc = EntanglementForgedGroundStateSolver(
            converter,
            self.mock_ts_ansatz,
            bitstrings_u=bitstrings_u,
            bitstrings_v=bitstrings_v,
            config=self.config,
            orbitals_to_reduce=[],
        )
        res = calc.solve(problem)
        self.assertAlmostEqual(-267.51696723002567, res.ground_state_energy)

    def test_TS_3(self):
        driver = EntanglementForgedDriver(
            hcore=self.hcore_ts,
            mo_coeff=np.eye(4, 4),
            eri=self.eri_ts,
            num_alpha=2,
            num_beta=2,
            nuclear_repulsion_energy=self.energy_shift_ts,
        )
        problem = ElectronicStructureProblem(driver)
        problem.second_q_ops()

        converter = QubitConverter(JordanWignerMapper())

        bitstrings_u = [
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
        ]
        bitstrings_v = [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
        ]

        calc = EntanglementForgedGroundStateSolver(
            converter,
            self.mock_ts_ansatz,
            bitstrings_u=bitstrings_u,
            bitstrings_v=bitstrings_v,
            config=self.config,
            orbitals_to_reduce=[],
        )
        res = calc.solve(problem)
        self.assertAlmostEqual(-267.5203802563949, res.ground_state_energy)

    def test_TS_4(self):
        driver = EntanglementForgedDriver(
            hcore=self.hcore_ts,
            mo_coeff=np.eye(4, 4),
            eri=self.eri_ts,
            num_alpha=2,
            num_beta=2,
            nuclear_repulsion_energy=self.energy_shift_ts,
        )
        problem = ElectronicStructureProblem(driver)
        problem.second_q_ops()

        converter = QubitConverter(JordanWignerMapper())

        bitstrings_u = [[1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 1, 0]]
        bitstrings_v = [[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0]]

        calc = EntanglementForgedGroundStateSolver(
            converter,
            self.mock_ts_ansatz,
            bitstrings_u=bitstrings_u,
            bitstrings_v=bitstrings_v,
            config=self.config,
            orbitals_to_reduce=[],
        )
        res = calc.solve(problem)
        self.assertAlmostEqual(-267.49261903008824, res.ground_state_energy)

    def test_O2_1(self):
        driver = EntanglementForgedDriver(
            hcore=self.hcore_o2,
            mo_coeff=np.eye(8, 8),
            eri=self.eri_o2,
            num_alpha=6,
            num_beta=6,
            nuclear_repulsion_energy=self.energy_shift_o2,
        )
        problem = ElectronicStructureProblem(driver)
        problem.second_q_ops()

        converter = QubitConverter(JordanWignerMapper())

        bitstrings_u = [
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0, 1, 0],
            [1, 1, 1, 1, 0, 1, 1, 0],
            [1, 1, 0, 1, 1, 1, 1, 0],
        ]
        bitstrings_v = [
            [1, 1, 1, 1, 1, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 0, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 1, 1, 0],
        ]

        calc = EntanglementForgedGroundStateSolver(
            converter,
            self.mock_o2_ansatz,
            bitstrings_u=bitstrings_u,
            bitstrings_v=bitstrings_v,
            config=self.config,
            orbitals_to_reduce=[],
        )
        res = calc.solve(problem)
        self.assertAlmostEqual(-147.63645235088566, res.ground_state_energy)

    def test_CN(self):
        driver = EntanglementForgedDriver(
            hcore=self.hcore_cn,
            mo_coeff=np.eye(8, 8),
            eri=self.eri_cn,
            num_alpha=5,
            num_beta=4,
            nuclear_repulsion_energy=self.energy_shift_cn,
        )

        problem = ElectronicStructureProblem(driver)
        problem.second_q_ops()

        converter = QubitConverter(JordanWignerMapper())

        bitstrings_u = [
            [1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 1, 0, 1, 0],
            [1, 1, 0, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 1, 1, 0, 0],
            [1, 1, 0, 1, 1, 0, 1, 0],
            [1, 1, 1, 1, 1, 0, 0, 0],
        ]
        bitstrings_v = [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 1, 0, 0],
            [1, 1, 1, 0, 0, 1, 0, 0],
            [1, 1, 0, 1, 0, 0, 1, 0],
            [1, 0, 1, 1, 1, 0, 0, 0],
        ]

        calc = EntanglementForgedGroundStateSolver(
            converter,
            self.mock_cn_ansatz,
            bitstrings_u=bitstrings_u,
            bitstrings_v=bitstrings_v,
            config=self.config,
            orbitals_to_reduce=[],
        )
        res = calc.solve(problem)
        self.assertAlmostEqual(-91.0419294719206, res.ground_state_energy)

    def test_CH3(self):
        driver = EntanglementForgedDriver(
            hcore=self.hcore_ch3,
            mo_coeff=np.eye(6, 6),
            eri=self.eri_ch3,
            num_alpha=3,
            num_beta=2,
            nuclear_repulsion_energy=self.energy_shift_ch3,
        )

        problem = ElectronicStructureProblem(driver)
        problem.second_q_ops()

        converter = QubitConverter(JordanWignerMapper())

        bitstrings_u = [
            [1, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 1],
            [1, 0, 1, 0, 1, 0],
            [1, 0, 1, 1, 0, 0],
            [0, 1, 1, 1, 0, 0],
        ]
        bitstrings_v = [
            [1, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0],
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0],
        ]

        calc = EntanglementForgedGroundStateSolver(
            converter,
            self.mock_ch3_ansatz,
            bitstrings_u=bitstrings_u,
            bitstrings_v=bitstrings_v,
            config=self.config,
            orbitals_to_reduce=[],
        )
        res = calc.solve(problem)
        self.assertAlmostEqual(-39.09031477502881, res.ground_state_energy)
