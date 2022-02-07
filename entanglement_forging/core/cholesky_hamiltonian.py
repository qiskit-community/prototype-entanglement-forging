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

"""Cholesky hamiltonian module."""

import numpy as np
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.problems.second_quantization.electronic.builders.fermionic_op_builder import \
    build_ferm_op_from_ints
from qiskit_nature.properties.second_quantization.electronic.integrals import IntegralProperty, OneBodyElectronicIntegrals, TwoBodyElectronicIntegrals
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis

from ..utils.log import Log


# pylint: disable=invalid-name
def modified_cholesky(two_body_overlap_integrals, eps):
    """ Performs modified Cholesky decomposition ("mcd") on array
    two_body_overlap_integrals with threshold eps.

    H= H_1 \\otimes I + I \\otimes H_1 + \\sum_\\gamma L_\\gamma \\otimes L_\\gamma

    Parameters:
    two_body_overlap_integrals (np.ndarray): A 4D array containing values of all
                                             2-body h2 overlap integrals
    eps (float): The threshold for the decomposition (typically a number close to 0)

    Returns:
    A tuple containing:
        n_gammas (int): the number of Cholesky matrices L_\\gamma
        L (numpy.ndarray): a 3D array containing the set of Cholesky matrices L_\\gamma
    """
    n_basis_states = two_body_overlap_integrals.shape[0]  # number of basis states
    # max number of Cholesky vectors, and current number of Cholesky vectors
    # (n_gammas = "number of gammas")
    chmax, n_gammas = 10 * n_basis_states, 0
    W = two_body_overlap_integrals.reshape(n_basis_states ** 2, n_basis_states ** 2)
    L = np.zeros((n_basis_states ** 2, chmax))
    Dmax = np.diagonal(W).copy()
    nu_max = np.argmax(Dmax)
    vmax = Dmax[nu_max]
    while vmax > eps:
        L[:, n_gammas] = W[:, nu_max]
        if n_gammas > 0:
            L[:, n_gammas] -= np.dot(L[:, 0:n_gammas], L.T[0:n_gammas, nu_max])
        L[:, n_gammas] /= np.sqrt(vmax)
        Dmax[:n_basis_states ** 2] -= L[:n_basis_states ** 2, n_gammas] ** 2
        n_gammas += 1
        nu_max = np.argmax(Dmax)
        vmax = Dmax[nu_max]
    L = L[:, :n_gammas].reshape((n_basis_states, n_basis_states, n_gammas))
    return n_gammas, L


def get_fermionic_ops_with_cholesky(
        mo_coeff,
        h1,
        h2,
        opname,
        halve_transformed_h2=False,
        occupied_orbitals_to_reduce=None,
        virtual_orbitals_to_reduce=None,
        epsilon_cholesky=1e-10,
        verbose=False):
    """ Decomposes the Hamiltonian operators into a form appropriate for entanglement forging.

    Parameters:
    mo_coeff (np.ndarray): 2D array representing coefficients for converting from AO to MO basis.
    h1 (np.ndarray): 2D array representing operator
                     coefficients of one-body integrals in the AO basis.
    h2 (np.ndarray or None): 4D array representing operator coefficients
                             of two-body integrals in the AO basis.
    halve_transformed_h2 (Bool): Optional; Should be set to True for Hamiltonian
                                 operator to agree with Qiskit conventions, apparently.
    occupied_orbitals_to_reduce (list): Optional; A list of occupied orbitals that will be removed.
    virtual_orbitals_to_reduce (list):Optional; A list of virtual orbitals that will be removed.
    epsilon_cholesky (float): Optional; The threshold for the decomposition
                              (typically a number close to 0).
    verbose (Bool): Optional; Option to print various output during computation.

    Returns:
    qubit_op (WeightedPauliOperator): H_1 in the Cholesky decomposition.
    cholesky_ops (list of WeightedPauliOperator): L_\\gamma in the Cholesky decomposition
    freeze_shift (float): Energy shift due to freezing.
    h1 (numpy.ndarray): 2D array representing operator coefficients of one-body
                        integrals in the MO basis.
    h2 (numpy.ndarray or None): 4D array representing operator coefficients of
                                two-body integrals in the MO basis.
    """
    if virtual_orbitals_to_reduce is None:
        virtual_orbitals_to_reduce = []
    if occupied_orbitals_to_reduce is None:
        occupied_orbitals_to_reduce = []
    C = mo_coeff
    del mo_coeff
    h1 = np.einsum('pi,pr->ir', C, h1)
    h1 = np.einsum('rj,ir->ij', C, h1)  # h_{pq} in MO basis

    # do the Cholesky decomposition:
    if h2 is not None:
        ng, L = modified_cholesky(h2, epsilon_cholesky)
        if verbose:
            h2_mcd = np.einsum('prg,qsg->prqs', L, L)
            Log.log("mcd threshold =", epsilon_cholesky)
            Log.log(
                "deviation between mcd and original eri =",
                np.abs(
                    h2_mcd -
                    h2).max())
            Log.log("number of Cholesky vectors =", ng)
            Log.log("L.shape = ", L.shape)
            del h2_mcd
        # obtain the L_{pr,g} in the MO basis
        L = np.einsum('prg,pi,rj->ijg', L, C, C)
    else:
        size = len(h1)
        ng, L = 0, np.zeros(shape=(size, size, 0))

    if occupied_orbitals_to_reduce is None:
        occupied_orbitals_to_reduce = []

    if virtual_orbitals_to_reduce is None:
        virtual_orbitals_to_reduce = []

    if len(occupied_orbitals_to_reduce) > 0:
        if verbose:
            Log.log('Reducing occupied orbitals:', occupied_orbitals_to_reduce)
        orbitals_not_to_reduce = list(
            sorted(set(range(len(h1))) - set(occupied_orbitals_to_reduce)))

        h1_frozenpart = h1[np.ix_(
            occupied_orbitals_to_reduce, occupied_orbitals_to_reduce)]
        h1_activepart = h1[np.ix_(
            orbitals_not_to_reduce, orbitals_not_to_reduce)]
        L_frozenpart = L[np.ix_(
            occupied_orbitals_to_reduce, occupied_orbitals_to_reduce)]
        L_activepart = L[np.ix_(orbitals_not_to_reduce,
                                orbitals_not_to_reduce)]

        freeze_shift = 2 * np.einsum('pp', h1_frozenpart) \
                       + 2 * np.einsum('ppg,qqg', L_frozenpart, L_frozenpart) \
                       - np.einsum('pqg,qpg', L_frozenpart, L_frozenpart)
        h1 = h1_activepart + 2 * np.einsum('ppg,qsg->qs', L_frozenpart, L_activepart) \
             - np.einsum('psg,qpg->qs',
                         L[np.ix_(occupied_orbitals_to_reduce, orbitals_not_to_reduce)],
                         L[np.ix_(orbitals_not_to_reduce, occupied_orbitals_to_reduce)])
        L = L_activepart
    else:
        freeze_shift = 0

    if len(virtual_orbitals_to_reduce) > 0:
        if verbose:
            Log.log('Reducing virtual orbitals:', virtual_orbitals_to_reduce)
        virtual_orbitals_to_reduce = np.asarray(virtual_orbitals_to_reduce)
        virtual_orbitals_to_reduce -= len(occupied_orbitals_to_reduce)
        orbitals_not_to_reduce = list(
            sorted(set(range(len(h1))) - set(virtual_orbitals_to_reduce)))
        h1 = h1[np.ix_(orbitals_not_to_reduce, orbitals_not_to_reduce)]
        L = L[np.ix_(orbitals_not_to_reduce, orbitals_not_to_reduce)]
    else:
        pass

    h2 = np.einsum('prg,qsg->prqs', L, L)
    if halve_transformed_h2:
        h2 /= 2
    h1_int = OneBodyElectronicIntegrals(basis=ElectronicBasis.SO, matrices=h1)
    h2_int = TwoBodyElectronicIntegrals(basis=ElectronicBasis.SO, matrices=h2)
    int_property = IntegralProperty("fer_op", [h1_int, h2_int])

    fer_op = int_property.second_q_ops()["fer_op"]

    converter = QubitConverter(JordanWignerMapper())
    qubit_op = converter.convert(fer_op)
    qubit_op._name = opname + '_onebodyop'  # pylint: disable=protected-access
    cholesky_ops = []
    for g in range(L.shape[2]):
        cholesky_int = OneBodyElectronicIntegrals(basis=ElectronicBasis.SO, matrices=L[:, :, g])
        cholesky_property = IntegralProperty("cholesky_op", [cholesky_int])
        cholesky_op = converter.convert(cholesky_property.second_q_ops()["cholesky_op"])
        cholesky_op._name = opname + '_chol' + str(g)  # pylint: disable=protected-access
        cholesky_ops.append(cholesky_op)
    return qubit_op, cholesky_ops, freeze_shift, h1, h2
