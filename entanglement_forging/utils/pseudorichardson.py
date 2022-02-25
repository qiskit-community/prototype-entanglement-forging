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

"""Pseudo Richardson circuits module."""
import numpy as np


def make_pseudorichardson_circuits(transpiled_circuits, simple_richardson_orders=None):
    """Creates pseudo Richardson circuits.

    Args:
        transpiled_circuits: Transpiled circuits
        simple_richardson_orders: Richardson orders

    Returns:
        List of pseudo Richardson circuits.
    """
    if simple_richardson_orders is None:
        simple_richardson_orders = [0]
    basic_insts = ["measure", "reset", "barrier", "snapshot"]
    final_circuits = []
    for qcirc in transpiled_circuits:
        for order in simple_richardson_orders:
            stretch_factor = 2 * (order) + 1
            name_parts = qcirc.name.split("_")
            new_qc = qcirc.copy(
                name="_".join(
                    name_parts[:-1]
                    + [f"richardson{stretch_factor:.2f}", name_parts[-1]]
                )
            )
            new_qc._data = []  # pylint: disable=protected-access
            # already handled in qcirc.copy()?
            new_qc._layout = qcirc._layout  # pylint: disable=protected-access
            for instr in qcirc.data:
                (operator, qargs, cargs) = instr
                new_qc.append(operator, qargs=qargs, cargs=cargs)
                if order == 0 or operator.name in basic_insts:
                    continue
                if operator.name in ["delay"]:
                    op_inv = operator
                else:
                    op_inv = operator.inverse()
                for _ in range(order):
                    new_qc.barrier(qargs)
                    # pylint: disable=expression-not-assigned
                    if operator.name == "sx":
                        [new_qc.rz(np.pi, q) for q in qargs]
                        [new_qc.sx(q) for q in qargs]
                        [new_qc.rz(np.pi, q) for q in qargs]
                    else:
                        new_qc.append(op_inv, qargs=qargs, cargs=cargs)
                    # pylint: enable=expression-not-assigned
                    new_qc.barrier(*qargs)
                    new_qc.append(operator, qargs=qargs, cargs=cargs)
            final_circuits.append(new_qc)
    return final_circuits


# pylint: disable=too-many-locals
def richardson_extrapolate(ydata, stretch_factors, axis, max_polyfit_degree=None):
    """Richardson extrapolation.

    Args:
        ydata: a numpy array of expectation values acquired at different stretch factors.
            The last axis of ydata should have length 2,
            with the 0th element being the expectation value,
            and the next element being the standard deviation.
        stretch_factors:
        axis: which axis of ydata corresponds to the stretch_factors.
        max_polyfit_degree: Extrapolation will proceed by (effectively) fitting
            a polynomial to the results from the different stretch factors.
            If max_polyfit_degree is None, this polynomial is degree len(stretch_factors)-1,
            so e.g. if there are only 2 stretch factors then a linear fit is used.
            If max_polyfit_degree is an integer less than len(stretch_factors)-1,
            the polynomial will be constrained to that degree.
            E.g. if you have 3 stretch factors but wanted to fit just a line,
            set max_polyfit_degree = 1.


    Returns:
        ydata_corrected: an array with shape like ydata but with the axis 'axis' eliminated.
    """

    polyfit_degree = len(stretch_factors) - 1
    if max_polyfit_degree:
        polyfit_degree = min(polyfit_degree, max_polyfit_degree)

    indexing_each_axis = [slice(None)] * ydata.ndim
    if polyfit_degree == 0:
        indexing_each_axis[axis] = 0
        ydata_corrected = ydata[tuple(indexing_each_axis)]
    elif polyfit_degree == 1 and len(stretch_factors) == 2:
        # faster/vectorized implementation of the case I care about
        # right now (TODO: generalize this) instead of calling curve_fit/polyfit:  # pylint: disable=fixme
        stretch1, stretch2 = stretch_factors
        denom = stretch2 - stretch1
        indexing_each_axis[-1] = 0
        indexing_each_axis[axis] = 0
        y1 = ydata[tuple(indexing_each_axis)]  # pylint: disable=invalid-name
        indexing_each_axis[axis] = 1
        y2 = ydata[tuple(indexing_each_axis)]  # pylint: disable=invalid-name
        indexing_each_axis[-1] = 1
        indexing_each_axis[axis] = 0
        var1 = ydata[tuple(indexing_each_axis)]
        indexing_each_axis[axis] = 1
        var2 = ydata[tuple(indexing_each_axis)]
        y_extrap = (
            y1 * stretch2 - y2 * stretch1
        ) / denom  # pylint: disable=invalid-name
        var_extrap = var1 * (stretch2 / denom) ** 2 + var2 * (stretch1 / denom) ** 2
        ydata_corrected = np.stack([y_extrap, var_extrap], axis=-1)
    else:
        raise NotImplementedError(
            "TODO: implement general Richardson extrapolation using numpy Polynomial.fit() "
            "inside loop, or else (maybe better?) vectorized general matrix equation"
        )

    return ydata_corrected
