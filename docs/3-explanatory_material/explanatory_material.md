# Explanatory material for the entanglement forging module

## Table of Contents

1. <a href="#overview-of-entanglement-forging">Overview of entanglement forging</a>
2. <a href="#entanglement-forging-procedure">Entanglement Forging Procedure</a>
3. <a href="#scaling">Scaling</a>
    - <a href="#freezing-orbitals">Freezing orbitals</a> 
    - <a href="#picking-the-bitstrings">Picking the bitstrings</a>
    - <a href="#designing-the-ansatz-used-in-entanglement-forging">Designing the ansatz used in Entanglement Forging</a>
    - <a href="#picking-the-backend">Picking the backend</a>
4. <a href="#%EF%B8%8F-current-limitations">⚠️ Current limitations</a>    
    - <a href="#ansatz--bitstrings">Ansatz & bitstrings</a>
    - <a href="#orbitals">Orbitals</a>    
    - <a href="#converter">Converter</a> 
    - <a href="#running-on-quantum-hardware">Running on quantum hardware</a>
    - <a href="#unsupported-qiskit-vqe-features">Unsupported Qiskit VQE features</a>    
5. <a href="#troubleshooting">Troubleshooting</a>    
    - <a href="#getting-poor-results-on-the-hardware">Getting poor results on the hardware</a> 
    - <a href="#for-ibm-power-users">For IBM Power users</a>    
6. <a href="#references">References</a>   

## Overview of entanglement forging

Entanglement forging [1] was introduced as a way to reduce the number of qubits necessary to perform quantum simulation of chemical or physical systems. In general, to simulate *n* orbitals in a chemistry problem, one typically needs 2*n* qubits. Entanglement Forging makes it possible to represent expectation values of a 2*n*-qubit wavefunction as sums of multiple expectation values of *n*-qubit states, embedded in a classical computation, thus doubling the size of the system that can be *exactly* simulated with a fixed number of qubits. Furthermore, Entanglement Forging permits the circuits necessary for the *n*-qubit simulations to be shallower, relaxing requirements on gate error and connectivity, at the cost of increased quantum and classical run times.

Previous techniques for reducing qubit count in quantum simulation applications could either reduce qubits slightly at the expense of deeper circuits (e.g. 2-qubit reduction, tapering), or yield a 50% qubit reduction at the expense of lower accuracy (e.g. restricted simulations). Using Entanglement Forging, one can achieve a 50% reduction in the number of qubits without compromising accuracy.

The underlying idea which enables Entanglement Forging is that a quantum system on 2*n* qubits can be partitioned into 2 subsystems, and that a Schmidt decomposition of the 2*n*-qubit wavefunction with respect to those subsystems is possible. Because of this decomposition, we obtain an accurate classical representation of the entanglement between the two subsystems.

The schematic below outlines how the expectation value *M* of a 2*n*-qubit wavefunction |ψ><sub>2*n*</sub> with respect to a 2*n*-qubit Hamiltonian H<sub>2*n*</sub> can be decomposed into a sum of expectation values of products of *n*-qubit wavefunctions with respect to *n*-qubit operators. These *n*-qubit expectation values correspond to sub-experiments.

![Entanglement Forging Infographic](figs/forging_info_graphic.png)

## Entanglement Forging Procedure

Entanglement Forging leverages near-term, heuristic algorithms, such as VQE, to provide an estimate of the 2*n*-qubit expectation value. It does so by assuming a parameterized ansatz for the wavefunction of each sub-system. (Note that the parameters of this ansatz describe the unitaries *U* and *V* in the Schmidt decomposition.) After the expectation value *M* has been decomposed into sub-experiments, the procedure is as follows:
1. Execute each sub-experiment on the QPU a number of times necessary to obtain sufficient statistics.
2. Combine the expectation values for the sub-experiments with the weights *w*<sub>a,b</sub> and the Schmidt parameters *λ*<sub>n</sub> to obtain an estimate for *M*.
3. Send the estimate of *M*, along with *λ*<sub>n</sub> and the variational parameters {θ} describing *U* and *V*, to a classical optimizer.
4. Use the classical optimizer to further minimize *M* and provide a new set for the variational parameters {θ} and Schmidt coefficients *λ*<sub>n</sub>.
5. Update the sub-experiments based on the updated {θ} and *λ*<sub>n</sub>.
6. Repeat Steps 1-5 until the estimate for *M* converges.

Note that if *M* is the expectation value of the system's Hamiltonian, then it is possible to separate the optimization over the variational parameters {θ} and the Schmidt coefficients *λ*<sub>n</sub>. In particular, the Schmidt coefficients can be optimized after step 2, and separately from the variational parameters.

Further, an easy way to reduce the number of sub-experiments necessary is by truncating the Schmidt decomposition of |ψ> to include only some number of the bitstring states |*b*<sub>n</sub>>. However, doing so will generally lead to less accuracy in the estimation of the expectation value.

## Scaling

The execution time scales differently with various properties of the simulations, and is indicated in the table below.

| Quantity | Scaling | Notes | Ways to Reduce |
| --- | ---- | --- |  --- |
| Orbitals | Fifth power | | [Orbital freezing](#freezing-orbitals) |
| Bitstring states \|*b*<sub>n</sub>>| Quadratic| Increasing the number of bitstring states  can increase the accuracy of the simulation, but at the expense of execution time. | [Schmidt decomposition truncation](#picking-the-bitstrings) |
| Ansatz parameters {θ} | Linear |An increased number of ansatz parameters can increase the accuracy of the simulation, but at the expense of execution time. | [Redesign the ansatz](#designing-the-ansatz-used-in-entanglement-forging) |

### Freezing orbitals

Since the execution time scales with the 5th power in the number of orbitals, it's a good idea to simplify the problem (if possible) by eliminating some of the orbitals. Some knowledge of chemistry is useful when picking orbitals to freeze. One good rule of thumb is to freeze the core orbital (for the case of water, this is the core oxygen 1s orbital). Furthermore, in the case of water, it turns out that orbital 3 (corresponding to the out-of-plane oxygen 2p orbitals) has different symmetry to the other orbitals, so excitations to orbital 3 are suppressed. For water, we thus freeze orbitals 0 and 3.

The core orbitals can be found with the following code:
```
qmolecule = driver.run()
print(f"Core orbitals: {qmolecule.core_orbitals}")
```

#### Example: Water molecule

The total number of orbitals (core + valence) = 7 orbitals

Frozen orbital approximation = 2 orbitals

Active space orbitals = total number of orbitals – frozen orbitals = 5 orbitals (bitstring size is set to 5)

Leading excitation analysis = 3 unique bitstrings

```python
>>> from entanglement_forging import reduce_bitstrings
>>> orbitals_to_reduce = [0,3]
>>> bitstrings = [[1,1,1,1,1,0,0],[1,0,1,1,1,0,1],[1,0,1,1,1,1,0]]
>>> reduced_bitstrings = reduce_bitstrings(bitstrings, orbitals_to_reduce)
>>> print(f'Bitstrings after orbital reduction: {reduced_bitstrings}')
Bitstrings after orbital reduction: [[1, 1, 1, 0, 0], [0, 1, 1, 0, 1], [0, 1, 1, 1, 0]]
```

### Picking the bitstrings

#### General Considerations

Picking appropriate bitstrings requires prior knowledge of the molecular electronic structure.

In general, the exact electronic wavefunction is a superposition of all possible distributions of the ~N~ electrons over the ~L~ orbitals and is exponential in size. However, only a relatively small number of excitations contribute significantly to the correlation energy. By identifying such leading electronic excitations, a linear combination of electronic configurations/Slater determinants that capture the most important portion of the Hilbert space and make the biggest contribution to the electronic wavefunction description can be selected. This allows for reduction in computational resources.

The leading electronic excitations can be represented in standard bitstrings (e.g. `[1,1,1,1,0,0,0]`). When an orbital is occupied by a spin up (α electron) or spin down (β electron), its bit will be set to 1. Therefore:
- the number of bits in each bitstring should be equal the number of spatial orbitals
- the number of 1s in each bitstring should equal the number of α or β particles.

Further reduction in computational resources can be achieved by [freezing some orbitals](./docs/4-explanatory_material/explanatory_material.md#freezing-orbitals) that do not participate in electronic excitations (i.e. core orbitals or those that lie out of symmetry) by removing the bits that correspond to them.

#### Fixing the Hartree-Fock bitstring

In some cases, it is possible to increase the accuracy of simulations and speed up the execution by setting `fix_first_bitstring=True` in `EntanglementForgedConfig`. This bypasses the computation of the first bitstring and replaces the result with HF energy.

This setting requires an ansatz that leaves the Hartree-Fock (HF) state unchanged under `var_form`. As a rule of thumb, this can be achieved by restricting entanglement between the qubits representing occupied orbitals (bits = 1) in the HF state and the qubits representing unoccupied orbitals (bits = 0) in the HF state.

For example, this figure from [1] shows the A, B, and C qubits entangled with the hop gates, D & E qubits entangled with hop gates, while the partition between (A,B,C) and (D,E) are only entangled with a CZ gate.

<img src="figs/Fig_5_c.png" width="250">

### Designing the ansatz used in Entanglement Forging

Because entanglement forging leverages a near-term, heuristic algorithm (namely, VQE), a judicious choice for the VQE ansatz can improve performance. Note that one way to design the ansatz is by endowing the unitaries *U* and *V* in the Schmidt decomposition with parameters. An open question is how to choose the best unitaries for a given problem.

For a chemistry simulation problem, the number of qubits in the circuit must equal the number of orbitals (minus the number of frozen orbitals, if applicable).

### Picking the backend

`backend` is an option in the [`EntanglementForgedConfig`](https://github.com/qiskit-community/prototype-entanglement-forging/blob/main/docs/2-reference_guide/reference_guide.md#options-entanglementforgedconfig) class. Users can choose between Statevector simulation, QASM simulation, or real quantum hardware. 

Statevector simulation is useful when we want to:
1. get the exact values of energies (e.g. for chemistry problems) without any error bars (assuming there are no other sources of randomness)
2. test the performance of an algorithm in the absence of shot noise (for VQE, there could be a difference between the trajectory of the parameters in the presence and absence of shot noise; in this case the statevector simulator can concretely provide an answer regarding the expressivity of a given ansatz without any uncertainty coming from shot noise)

QASM simulation is useful when:
1. the system sizes are larger because the statevector simulator scales exponentially in system size and will not be useful beyond small systems
2. simulating circuits with noise to mimic a real noisy quantum computer

When running the entanglement forging module either on the QASM simulator or on real quantum hardware, several additional options are available: `shots`, `bootstrap_trials`, `copysample_job_size`, `meas_error_mit`, `meas_error_shots`, `meas_error_refresh_period_minutes`, `zero_noise_extrap`. These options can be specified in the [`EntanglementForgedConfig`](https://github.com/qiskit-community/prototype-entanglement-forging/blob/main/docs/2-reference_guide/reference_guide.md#options-entanglementforgedconfig) class. Users can use the QASM simulator to test out these options before running them on real quantum hardware. 

Notes:
- In the limit of infinite shots, the mean value of the QASM simulator will be equal to the value of the statevector simulator.
- The QASM simulator also has a method that [mimics the statevector simulator](https://qiskit.org/documentation/tutorials/simulators/1_aer_provider.html) without shot noise as an alternative to statevector simulator.

## ⚠️ Current limitations

### Ansatz & bitstrings
- It is currently an open problem how to pick the best circuit (ansatze) for VQE (and thus Entanglement Forging) for a given system. 
- It is also currently an open problem how to pick the best bitstring for Entanglement Forging. 
- In the current implementation of the module, the spin-up and spin-down particles are treated symmetrically (U=V for the ansatz with equivalent bitstrings). 
  - It would be useful to be able to specify different bit strings for spin-up/spin down orbitals as this would open the door to open-shell species. 
  - There are plans in the future to break the ansatz and bitstring symmetry. This will be made possible after a new expectation values class is merged in Terra.
- In the current implementation of the module, the ansatz must be real. 
  - For molecular calculations, one can usually force the ansatz to be real. On the other hand, in crystalline solids (away from the gamma point and without inversion symmetry), the Hamiltonian is defined by the complex numbers.
  - There are plans in the future to implement complex ansatze. 

### Orbitals
- The current implementation of Forged VQE also requires that the number of alpha particles equals the number of beta particles. The relevant parameters can be found with the following code:
```
qmolecule = driver.run()
print(f"Number of spatial orbitals: {qmolecule.num_molecular_orbitals}")
print(f"Number of alpha particles: {qmolecule.num_alpha}")
print(f"Number of beta particles: {qmolecule.num_beta}")
```

### Converter
- The current implementation only supports the `JordanWignerMapper` converter. 

### Results
- In the current implementation, only the energy of the final state is available. It would be useful to have a feature to output the 1- and 2-body density matrices of the final state after the optimization.
  - The 1-body matrices are used for:
    - electrostatic properties
    - electronic densities
    - molecular electrostatic potential
  - 2-body matrices are used for:
    - orbital optimization
    - analysis of correlation functions
  - The combination of both is used in entanglement analysis.


### Running on quantum hardware
- Results on hardware will not be as good as on the QASM simulator. Getting good results will require using a quantum backend with good properties (qubit fidelity, gate fidelity etc.), as well as a lot of fine-tuning of parameters.
- Queue times are long, which makes execution on a quantum backend difficult. This issue will be mitigated once the module can be uploaded as a qiskit runtime (not currently supported). This will be made possible when either:
  - The module has been simplified to fit into a single program, or
  - Qiskit runtime provides support for multi-file programs

### Unsupported Qiskit VQE features

This module is based on Qiskit's Variational Quantum Eigensolver (VQE) algorithm, however, some of the features available in VQE are not currently supported by this module. Here is a list of known features that are not supported:

- Using `QuantumInstance` instead of `backend`.

This list is not exhaustive.

## Troubleshooting

### Getting poor results on the hardware

Try using `fix_first_bitstring=True` in `EntanglementForgedConfig`. This bypasses the computation of the first bitstring and replaces the result with the HF energy. This setting requires an ansatz that leaves the HF state unchanged under `var_form`.

### For IBM Power users

pip is not well-supported on IBM power, so everything should be installed with conda. To get the package to work on Power, one needs to:
- remove `matplotlib>=2.1,<3.4` from `requirements.txt`
- install matplotlib with conda manually instead of pip

```
pip uninstall matplotlib 
conda install matplotlib
```

## References

This module is based on the theory and experiment described in the following paper:

[1] Andrew Eddins, Mario Motta, Tanvi P. Gujarati, Sergey Bravyi, Antonio Mezzacapo, Charles Hadfield, Sarah Sheldon, *Doubling the size of quantum simulators by entanglement forging*, https://arxiv.org/abs/2104.10220
