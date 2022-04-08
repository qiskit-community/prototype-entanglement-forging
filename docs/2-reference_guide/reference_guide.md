# Reference guide for the entanglement forging module

## Table of Contents
1. [Installation instructions](#installation-instructions)
    - [Basic installation](#basic-installation)
    - [Installation from source](#installation-from-source)
2. [Using the module](#using-the-module)
    - [Specifying the problem](#specifying-the-problem)
    - [Specifying the bitstrings](#specifying-the-bitstrings)
    - [Freezing orbitals](#freezing-orbitals)
    - [Specifying the Ansatz](#specifying-the-ansatz)
    - [Options (`EntanglementForgedConfig`)](#options-entanglementforgedconfig)
    - [Specifying the converter](#specifying-the-converter)
    - [The solver](#the-solver)
    - [Running the algorithm](#running-the-algorithm)
    - [Viewing the results](#viewing-the-results)
    - [Verbose](#verbose)

This guide is for those who just want to use the package. If you want to extend the module or documentation, read [this other guide](CONTRIBUTING.md) instead. Installation instructions are only located here to avoid repetition.


## Installation instructions
:exclamation: _This prototype depends on the PySCF package, which does not support Windows; therefore, Windows users will not be able to install and use this software. Advanced Windows users may wish to attempt to install PySCF using Ubuntu via the Windows Subsystem for Linux.  We are exploring the possibility of providing Docker support for this prototype so it can be used within Docker Desktop, including on Windows._

Ensure your local environment is compatible with the entanglement-forging package:
  - Ensure you are on a supported operating system (macOS or Linux)
  - Ensure you are running a supported version of Python (py37,py38,py39)
  - (Optional) It can be useful to create a new environment (here called `my_forging_env`) and install Python 3.9 (recommended). There are several alternatives for this, using `conda` within the terminal:
    ```
    conda create -n my_forging_env python=3.9
    conda activate my_forging_env
    ```

### Basic installation
1. From the terminal, use pip to install the entanglement-forging package:
    ```
    pip install entanglement-forging
    ```
2. Users may now run the entanglement forging demo notebooks on their local machine or use the entanglement-forging package in their own software.

### Installation from source
0. Make sure you have [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and [pip](https://pip.pypa.io/en/stable/installation/) (and optionally [Miniconda](https://docs.conda.io/en/latest/miniconda.html)) installed.
1. From the terminal, clone repository:
    ```
    git clone https://github.com/qiskit-community/prototype-entanglement-forging.git
    ```
    Alternatively, instead of cloning the original repository, you may choose to clone your personal [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo). You can do so by using the appropriate URL and adding the original repo to the list of remotes (here under the name `upstream`). This will be requiered for contribution unless you are granted write permissions for the original repository.
    ```
    git clone <YOUR-FORK-URL>
    git remote add upstream https://github.com/qiskit-community/prototype-entanglement-forging.git
    ```
2. Change directory to the freshly cloned forging module:
    ```
    cd prototype-entanglement-forging
    ```
3. Install the dependencies needed:
    ```
    pip install -r requirements.txt
    ```
4. (Optional) Install the developer dependencies:
    ```
    pip install -r dev-requirements.txt
    ```


## Using the module

### Specifying the problem
The module supports two options to specify the problem.

_Option 1_: with the `ElectronicStructureProblem` object from Qiskit.
  ```python
  problem = ElectronicStructureProblem(
    PySCFDriver(molecule=Molecule(geometry=[('H', [0., 0., 0.]),
                                            ('H', [0., 0., 0.735])],
                                  charge=0, multiplicity=1),
                basis='sto3g')
  )
  ```

_Option 2_: specifying the properties of the system directly to the `EntanglementForgedDriver` object.
  ```python
  # Coefficients that define the one-body terms of the Hamiltonian
  hcore = np.array([
      [-1.12421758, -0.9652574],
      [-0.9652574,- 1.12421758]
  ])

  # Coefficients that define the two-body terms of the Hamiltonian
  mo_coeff = np.array([
      [0.54830202, 1.21832731],
      [0.54830202, -1.21832731]
  ])

  # Coefficients for the molecular orbitals
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
  ```

The second option is useful when:
  1. you don't want to study a molecule (situations where there is no driver, so you want to feed the Hamiltonian by by-passing pyscf driver
  2. you want to manipulate the electronic structure of the system in a way that is not supported by the driver (molecular, but not in standard tool kit)

### Specifying the bitstrings
Bitstrings are specified as a list of lists.

_Example 1_: Two qubits with two bitstrings.
  ```python
  bitstrings = [[1,0],[0,1]]
  ```
_Example 2_: Seven qubits with three bitstrings.
  ```python
  bitstrings = bitstrings = [[1,1,1,1,1,0,0],[1,0,1,1,1,0,1],[1,0,1,1,1,1,0]]
  ```

For information on picking bitstrings, refer to [this section](/docs/3-explanatory_material/explanatory_material.md#picking-the-bitstrings) of the Explanatory Material.

For current limitations on specifying bitstrings, refer to [this section](/docs/3-explanatory_material/explanatory_material.md#ansatz--bitstrings) of the Explanatory Material.

### Freezing orbitals
Orbitals can be frozen by specifying them as a list, then using `reduce_bitstrings`.

_Example_: freezing the zeroth orbital (ground state) and the third orbital.
  ```python
  orbitals_to_reduce = [0,3]
  bitstrings = [[1,1,1,1,1,0,0],[1,0,1,1,1,0,1],[1,0,1,1,1,1,0]]
  reduced_bitstrings = reduce_bitstrings(bitstrings, orbitals_to_reduce)
  ```
_Output_:
  ```python
  >>> reduced_bitstrings
  [1,1,1,0,0],[0,1,1,0,1],[0,1,1,1,0]]
  ```

For discussion of scaling and orbital freezing, refer to the [this section](/docs/3-explanatory_material/explanatory_material.md#freezing-orbitals) of the Explanatory Material.

### Specifying the Ansatz
The module supports two options to specify the ansatz.

_Option 1_: Using a parameterized circuit.
  ```python
  from qiskit.circuit.library import TwoLocal

  ansatz = TwoLocal(2, [], 'cry', [[0,1],[1,0]], reps=1)
  ```

_Option 2_: Using parametrized gates.
  ```python
  from qiskit.circuit import Parameter, QuantumCircuit

  theta_0, theta_1 = Parameter('θ0'), Parameter('θ1')

  ansatz = QuantumCircuit(2)
  ansatz.cry(theta_0, 0, 1)
  ansatz.cry(theta_1, 1, 0)
  ```

### Options (`EntanglementForgedConfig`)
`EntanglementForgedConfig` contains all of the options for running the algorithm. It can be defined as follows:
```
config = EntanglementForgedConfig(backend=backend, maxiter = 200, initial_params=[0,0.5*np.pi])
```

The options are:
  - `backend`: Instance of selected backend.
  - `qubit_layout`: Initial position of virtual qubits on physical qubits. If this layout makes the circuit compatible with the coupling_map constraints, it will be used.
  - `initial_params` (NoneType or list of int): A list specifying the initial optimization parameters.
  - `maxiter` (int): Maximum number of optimizer iterations to perform.
  - `optimizer_name` (str): e.g. 'SPSA', 'ADAM', 'NELDER_MEAD', 'COBYLA', 'L_BFGS_B', 'SLSQP' ...
  - `optimizer_tol` (float): Optimizer tolerance, e.g. 1e-6.
  - `skip_any_optimizer_cal` (bool): Setting passed to any optimizer with a 'skip_calibration' argument.
  - `spsa_last_average` (int): Number of times to average over final SPSA evaluations to determine optimal parameters (only used for SPSA).
  - `initial_spsa_iteration_idx` (int): Iteration index to resume interrupted VQE run (only used for SPSA).
  - `spsa_c0` (float): The initial parameter 'a'. Determines step size to update parameters in SPSA (only used for SPSA).
  - `spsa_c1` (float): The initial parameter 'c'. The step size used to approximate gradient in SPSA (only used for SPSA).
  - `max_evals_grouped` (int): Maximum number of evaluations performed simultaneously.
  - `rep_delay` (float): Delay between programs in seconds.
  - `shots` (int): The total number of shots for the simulation (overwritten for the statevector backend).
  - `fix_first_bitstring` (bool): Bypasses computation of first bitstring and replaces result with HF energy. This setting assumes that the first bitstring is the HF state (e.g. [1,1,1,1,0,0,0]). Can speed up the computation, but requires ansatz that leaves the HF state unchanged under var_form.
  - `bootstrap_trials` (int): A setting for generating error bars (not used for the statevector backend).
  - `copysample_job_size` (int or NoneType): A setting to approximately realize weighted sampling of circuits according to their relative significance
          (Schmidt coefficients). This number should be bigger than the number of unique circuits running (not used for the statevector backend).
  - `meas_error_mit` (bool): Performs measurement error mitigation (not used for the statevector backend).
  - `meas_error_shots` (int): The number of shots for measurement error mitigation (not used for the statevector backend).
  - `meas_error_refresh_period_minutes` (float): How often to refresh the calibration
          matrix in measurement error mitigation, in minutes (not used for the statevector backend).
  - `zero_noise_extrap` (bool): Linear extrapolation for gate error mitigation (ignored for the statevector backend)

### Specifying the converter
The qubit converter is specified as:
```
converter = QubitConverter(JordanWignerMapper())
```

### The solver
`EntanglementForgedGroundStateSolver` is the ground state calculation interface for entanglement forging. It is specified as follows:
```
forged_ground_state_solver = EntanglementForgedGroundStateSolver(converter, ansatz, bitstrings, config)
```

### Running the algorithm
```
forged_result = forged_ground_state_solver.solve(problem)
```

### Viewing the results
The results can be viewed by calling `forged_result`:
```
>>> forged_result
Ground state energy (Hartree): -1.1372604047327592
Schmidt values: [-0.99377311  0.11142266]
Optimizer parameters: [  6.2830331  -18.77969966]
```

### Verbose
To activate verbose output
```
from entanglement_forging import Log
Log.VERBOSE = True
```
