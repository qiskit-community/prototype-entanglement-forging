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

"""VQE results"""

import inspect
import pprint
from collections import OrderedDict
from typing import List, Optional, Union, Tuple

import numpy as np
from qiskit.algorithms.minimum_eigen_solvers.vqe import VQEResult


# pylint: disable=too-few-public-methods,too-many-arguments
class AuxiliaryResults:
    """Base class for auxiliary results."""

    def __str__(self) -> str:
        result = OrderedDict()
        for name, value in inspect.getmembers(self):
            if (
                not name.startswith("_")
                and not inspect.ismethod(value)
                and not inspect.isfunction(value)
                and hasattr(self, name)
            ):
                result[name] = value

        return pprint.pformat(result, indent=4)

    def __repr__(self):
        key_value_pairs = [
            f"{name}: {value}"
            for name, value in inspect.getmembers(self)
            if not name.startswith("_")
            and not inspect.ismethod(value)
            and not inspect.isfunction(value)
            and hasattr(self, name)
        ]
        return f"{self.__class__.__name__}({key_value_pairs})"


class OptimalParams(AuxiliaryResults):
    """Optional params results."""

    def __init__(self, energy: float, optimal_params: List[float]):
        """Optimal parameters.

        Args:
            energy: energy hartree
            optimal_params: optimal parameters for VQE
        """
        self.energy = energy
        self.optimal_params = optimal_params


class Bootstrap(AuxiliaryResults):
    """Bootstrap results."""

    def __init__(self, eval_count: int, eval_timestamp, parameters, bootstrap_values):
        """Bootstrap parameters.

        Args:
            eval_count:
            eval_timestamp:
            parameters:
            bootstrap_values:
        """
        self.eval_count = eval_count
        self.eval_timestamp = eval_timestamp
        self.parameters = parameters
        self.boostrap_values = bootstrap_values


class DataResults(AuxiliaryResults):
    """Data results."""

    def __init__(
        self,
        eval_count,
        eval_timestamp,
        energy_hartree,
        energy_std,
        parameters,
        schmidts,
    ):
        """Data results.

        Args:
            eval_count:
            eval_timestamp:
            energy_hartree:
            energy_std:
            parameters:
            schmidts:
        """
        self.eval_count = eval_count
        self.eval_timestamp = eval_timestamp
        self.energy_hartree = energy_hartree
        self.energy_str = energy_std
        self.parameters = parameters
        self.schmidts = schmidts


class EntanglementForgedVQEResult(VQEResult):
    """Entanglement-forged VQE Result."""

    def __init__(
        self,
        parameters_history: Optional[List[List[np.ndarray]]] = None,
        energies_history: Optional[List[List[np.ndarray]]] = None,
        schmidts_history: Optional[List[List[np.ndarray]]] = None,
        energy_std_each_parameter_set: Optional[List[Union[float, int]]] = None,
        energy_offset: Optional[float] = None,
        eval_count: Optional[int] = None,
        auxiliary_results: Optional[List[Tuple[str, AuxiliaryResults]]] = None,
    ) -> None:
        """Results for EntanglementForgedGroundStateSolver.

        Args:
            parameters_history:
            energies_history:
            schmidts_history:
            energy_std_each_parameter_set:
            energy_offset:
            eval_count:
            auxiliary_results: additional results (on order to remove writing to filesystem)
        """
        super().__init__()

        self._parameters_history = parameters_history or []
        self._energies_history = energies_history or []
        self._schmidts_history = schmidts_history or []

        self._energy_std_each_parameter_set = energy_std_each_parameter_set
        self._energy_offset = energy_offset
        self._eval_count = eval_count
        self.auxiliary_results = auxiliary_results

    def __repr__(self):
        return (
            "Ground state energy (Hartree): {gse}\n"
            "Schmidt values: {sv}\n"
            "Optimizer parameters: {op}".format(
                gse=self.ground_state_energy,
                sv=self.schmidts_value,
                op=self.optimizer_parameters,
            )
        )

    def get_parameters_history(self):
        """Returns a list of the optimizer parameters at each iteration."""
        return self._parameters_history

    def get_schmidts_history(self):
        """Returns a list of the Schmidt values at each iteration."""
        return self._schmidts_history

    def get_energies_history(self):
        """Returns a list of the energy at each iteration."""
        return [[j + self._energy_offset for j in i] for i in self._energies_history]

    @property
    def ground_state_energy(self):
        """Returns ground state energy."""
        return self.get_energies_history()[-1][0]

    @property
    def schmidts_value(self):
        """Returns Schmidts value."""
        return self._schmidts_history[-1][0]

    @property
    def optimizer_parameters(self):
        """Returns optimizer parameters."""
        return self._parameters_history[-1][0]

    @property
    def energy_offset(self) -> Optional[float]:
        """Returns energy offset."""
        return self._energy_offset

    @property
    def energy_std_each_parameter_set(self) -> Optional[List[Union[float, int]]]:
        """Returns energy std for each parameters set."""
        return self._energy_std_each_parameter_set

    @property
    def eval_count(self) -> Optional[int]:
        """Returns evaluation count."""
        return self._eval_count
