# Reference guide for the entanglement forging module

## Table of Contents

1. <a href="#contribution-guide">Contribution guide</a>
    - <a href="#initial-set-up-and-installing-dependencies">Initial set-up and installing dependencies</a>
    - <a href="#running-tests">Running tests</a>
    - <a href="#making-a-pull-request">Making a pull request</a>
2. <a href="#using-the-module">Using the module</a>
    - <a href="#installation-instructions">Installation instructions</a>
    - <a href="#specifying-the-problem">Specifying the problem</a>
    - <a href="#specifying-the-bitstrings">Specifying the bitstrings</a>
    - <a href="#freezing-orbitals">Freezing orbitals</a>
    - <a href="#specifying-the-ansatz">Specifying the Ansatz</a>
    - <a href="#options-entanglementforgedconfig">Options (`EntanglementForgedConfig`)</a>
    - <a href="#specifying-the-converter">Specifying the converter</a>
    - <a href="#the-solver">The solver</a>
    - <a href="#running-the-algorithm">Running the algorithm</a>
    - <a href="#viewing-the-results">Viewing the results</a>
    - <a href="#verbose">Verbose</a>

## Contribution guide

This guide is for those who want to extend the module or documentation. If you just want to use the package, skip to [this section](./reference_guide.md#using-the-module).

### Initial set-up and installing dependencies
0. Make sure you have [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and [pip](https://pip.pypa.io/en/stable/installation/) (and optionally [Miniconda](https://docs.conda.io/en/latest/miniconda.html)) installed.
1. Open terminal.

    1.5 (Optional) It can be useful to create a new environment (here called `my_forging_env`) and install Python 3.10 (recommended). Within terminal:
    ```
    conda create -n my_forging_env python=3.9
    ```
    then
    ```
    conda activate my_forging_env
    ```
2. Within terminal, clone repository:
```bash
git clone https://github.com/IBM-Quantum-Prototypes/entanglement-forging.git
```
3. Change directory to the freshly cloned forging module:
```bash
cd forging
```
4. Install the dependencies needed:
```
pip install -r requirements.txt
```

### Running tests
First install required packages:
```
pip install -r requirements-dev.txt
```
To run tests:
```
tox -e{env}
```
where you replace `{env}` with `py36`, `py37`, `py38` or `py39` depending on which version of python you have (to check python version, type `python --version` in the terminal).

To run all the tests:
```
python -m unittest discover -v tests
```

To run linting tests (checks formatting/syntax):
```
tox -elint
```

To run notebook tests (for more info, see [here](https://github.com/treebeardtech/nbmake)):
```
pip install pytest nbmake
pytest --nbmake **/*ipynb
```
Note: notebook tests check for execution and time-out errors, not correctness.

### Making a pull request

1. To make a contribution, first set up a remote branch (here called `my-contribution`) that is tracked:
```
git checkout main
git pull
git checkout -b my-contribution
```
... make your contribution now (edit some code, add some files) ...
```
git add .
git commit -m 'initial working version of my contribution'
git push -u origin my-contribution
```
2. Before making a Pull Request always get the latest changes from main:
```
git checkout main
git pull
git checkout my-contribution
git merge main
```
... fix any merge conflicts here ...
```
git add .
git commit -m 'merged updates from main'
git push
```
3. Go back to the `/forging` repo on _GitHub_, switch to your contribution branch (same name: `my-contribution`), and click "Pull Request". Write a clear explanation of the feature.
4. Under Reviewer, select Iskandar Sitdikov or Aggie Branczyk.
5. Click "Create Pull Request".
6. Your Pull Request will be reviewed and, if everything is ok, it will be merged.

## Using the module

### Installation instructions
0. Make sure you have [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and [pip](https://pip.pypa.io/en/stable/installation/) (and optionally [Miniconda](https://docs.conda.io/en/latest/miniconda.html)) installed.
1. Open terminal.

    1.5 (Optional) It can be useful to create a new environment (here called `my_forging_env`) and install Python 3.10 (recommended). Within terminal:
    ```
    conda create -n my_forging_env python=3.9
    ```
    then
    ```
    conda activate my_forging_env
    ```
2. Within terminal,  clone repository:
```bash
git clone https://github.com/IBM-Quantum-Prototypes/entanglement-forging.git
```
3. Change directory to the freshly cloned forging module:
```bash
cd forging
```
4. Install the package (this step gives you access to the module _outside_ of the cloned directory):
```bash
pip install -e .
```
5. You can now use the package.

### Specifying the problem

The module supports two options to specify the problem.

Option 1: with the `ElectronicStructureProblem` object from Qiskit.
```python
problem = ElectronicStructureProblem(
  PySCFDriver(molecule=Molecule(geometry=[('H', [0., 0., 0.]),
                                          ('H', [0., 0., 0.735])],
                                charge=0, multiplicity=1),
              basis='sto3g')
)
```

Option 2: specifying the properties of the system directly to the `EntanglementForgedDriver` object.
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

Example 1: Two qubits with two bitstrings.
```python
bitstrings = [[1,0],[0,1]]
```
Example 2: Seven qubits with three bitstrings.
```python
bitstrings = bitstrings = [[1,1,1,1,1,0,0],[1,0,1,1,1,0,1],[1,0,1,1,1,1,0]]
```

For information on picking bitstrings, refer to [this section](./docs/3-explanatory_material/explanatory_material.md#picking-the-bitstrings) of the Explanatory Material.

For current limitations on specifying bitstrings, refer to [this section](./docs/3-explanatory_material/explanatory_material.md#ansatz--bitstrings) of the Explanatory Material.

### Freezing orbitals

Orbitals can be frozen by specifying them as a list, then using `reduce_bitstrings`.

Example: freezing the zeroth orbital (ground state) and the third orbital.
```python
orbitals_to_reduce = [0,3]
bitstrings = [[1,1,1,1,1,0,0],[1,0,1,1,1,0,1],[1,0,1,1,1,1,0]]
reduced_bitstrings = reduce_bitstrings(bitstrings, orbitals_to_reduce)
```
Output:
```python
>>> reduced_bitstrings
[1,1,1,0,0],[0,1,1,0,1],[0,1,1,1,0]]
```

For discussion of scaling and orbital freezing, refer to the [this section](./docs/3-explanatory_material/explanatory_material.md#freezing-orbitals) of the Explanatory Material.

### Specifying the Ansatz

The module supports two options to specify the ansatz.

Option 1: Using a parameterized circuit.
```python
from qiskit.circuit.library import TwoLocal

ansatz = TwoLocal(2, [], 'cry', [[0,1],[1,0]], reps=1)
```

Option 2: Using parametrized gates.
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
