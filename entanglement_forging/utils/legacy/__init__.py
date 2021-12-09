# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Legacy Operators (:mod:`qiskit.aqua.operators.legacy`)
======================================================

.. currentmodule:: qiskit.aqua.operators.legacy

These are the Operators provided by Aqua up until the 0.6 release. These are being replaced
by the operator flow function and we encourage you to use this.

Note:
    At some future time this legacy operator logic will be deprecated and removed.

Legacy Operators
================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   TPBGroupedWeightedPauliOperator

"""

from .tpb_grouped_weighted_pauli_operator import TPBGroupedWeightedPauliOperator

__all__ = [
    'TPBGroupedWeightedPauliOperator'
]
