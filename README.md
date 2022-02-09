:exclamation: _This software is not compatible with the latest version of qiskit-terra. To avoid dependency conflicts, it is **highly recommended** that the dependencies for this software be installed within a fresh Python package management environment (e.g. [conda](https://docs.conda.io/en/latest/))._

# Entanglement forging module
This directory contains artefacts related to the entanglement forging module.

Table of Contents:
1. <a href="#the-module">The module</a>
2. <a href="#documentation">Documentation</a>
3. <a href="#how-to-give-feedback">How to give feedback</a>
4. <a href="#how-to-use-this-module">How to use this module</a>
    - <a href="#installation-instructions">Installation instructions</a>
    - <a href="#how-to-use-the-installed-package">How to use the installed package</a>
5. <a href="#for-contributors">For contributors</a>
    - <a href="#installing-dependencies">Installing dependencies</a>
    - <a href="#running-tests">Running tests</a>
6. <a href="#acknowledgments">Acknowledgments</a>
7. <a href="#references">References</a>
8. <a href="#license">License</a>

## The module
The module allows a user to simulate chemical and physical systems using a Variational Quantum Eigensolver (VQE) enhanced by Entanglement Forging  [[1]](./README.md#references). Entanglement Forging doubles the size of the system that can be *exactly* simulated on a fixed set of quantum bits.

The module contents can be found in the [entanglement_forging](./entanglement_forging/) directory.

## Documentation
The documentation for this module is structured around four functions:
- [Tutorials](./documentation/1-tutorials/): to teach a beginner user how the module works
- [A reference guide](./documentation/2-reference_guide/): technical description of the machinery of the module
- [Explanatory material](./documentation/3-explanatory_material/): discussion of concepts and further reading

The documentation can be found in the [documentation](./documentation/) directory.

Information about this documentation philosophy can be found on the [Diátaxis website](https://diataxis.fr/).

## How to give feedback
We encourage your feedback! You can share your thoughts with us by:
- [Opening an issue](https://github.com/IBM-Quantum-Prototypes/entanglement-forging/issues) in the repository
- [Starting a conversation on GitHub Discussions](https://github.com/IBM-Quantum-Prototypes/entanglement-forging/discussions)
- Filling out our [survey](https://airtable.com/shrFxJXYzjxf5tFvx)

## How to use this module

After [installing the module](./README.md#installation-instructions), we recommend that new users work their way through the examples in the [tutorials](./documentation/1-tutorials/). A brief overview of the technique can be found in [overview of Entanglement Forging](./documentation/3-explanatory_material/explanatory_material.md#overview-of-entanglement-forging) and [Entanglement Forging procedure](./documentation/3-explanatory_material/explanatory_material.md#entanglement-forging-procedure).

### ⚠️ CAUTION

Before using the module for new work, users should read through the [reference guide](./documentation/2-reference_guide/reference_guide.md) and the [explanatory material](./documentation/3-explanatory_material/explanatory_material.md), specifically the [current limitations](./documentation/3-explanatory_material/explanatory_material.md#%EF%B8%8F-current-limitations) of the module.

### Installation instructions
We suggest working inside a new conda environment with `python=3.8`.

Clone the repo and install the package as follows:
```bash
git clone https://github.com/IBM-Quantum-Prototypes/entanglement-forging.git
cd entanglement-forging
pip install -e .
```
You can now use the package.

For detailed instructions, including setting up a conda environment, refer to the [installation instructions](./documentation/2-reference_guide/reference_guide.md#installation-instructions).

### How to use the installed package

Now that you have the package installed, you can follow examples within [tutorial notebooks](./documentation/1-tutorials)
or see the simple example below for ground-state energy calculation of hydrogen (copy the code into a file `example.py` and execute within terminal using   `python example.py`):

```python
import numpy as np

from qiskit import Aer
from qiskit.circuit.library import TwoLocal
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.drivers import PySCFDriver, Molecule
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem

from entanglement_forging import (EntanglementForgedConfig,
                                  EntanglementForgedGroundStateSolver)
...

# specify problem
problem = ElectronicStructureProblem(
  PySCFDriver(molecule=Molecule(geometry=[('H', [0., 0., 0.]),
                                          ('H', [0., 0., 0.735])],
                                charge=0, multiplicity=1),
              basis='sto3g')
)

# specify parameters
bitstrings = [[1, 0], [0, 1]]
ansatz = TwoLocal(2, [], 'cry', [[0, 1], [1, 0]], reps=1)

# specify configuration for forgnig
config = EntanglementForgedConfig(backend=Aer.get_backend('statevector_simulator'),
                                  maxiter=30,
                                  initial_params=[0, 0.5 * np.pi])
# specify converter
converter = QubitConverter(JordanWignerMapper())

# create solver
forged_ground_state_solver = EntanglementForgedGroundStateSolver(
  converter, ansatz, bitstrings, config)

# run solver
forged_result = forged_ground_state_solver.solve(problem)

# get ground state energy
print(f"Ground state energy: {forged_result.ground_state_energy}.")
```

## For contributors

If you'd like to contribute to this module, please take a look at our
[contribution instructions](./documentation/2-reference_guide/reference_guide.md#contribution-guide). This project adheres to Qiskit's [code of conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

We use [GitHub issues](https://github.com/IBM-Quantum-Prototypes/entanglement-forging/issues) for tracking requests and bugs.

⚠️ Please note that we can't commit to implementing all feature requests, but we will review them and allocate resources when possible.

### Installing dependencies

To install the dependencies needed:
```
pip install -r requirements.txt
```
For detailed instructions, refer to the [setup instructions](./documentation/2-reference_guide/reference_guide.md#initial-set-up-and-installing-dependencies) in the [reference guide](./documentation/2-reference_guide/reference_guide.md).

### Running tests

See [here](./documentation/2-reference_guide/reference_guide.md#running-tests).

## Acknowledgments

This module is based on the theory and experiment described in [[1]](./README.md#references).

The initial code on which this module is based was written by Andrew Eddins, Mario Motta, Tanvi Gujarati, and Charles Hadfield. The module was developed by Aggie Branczyk, Iskandar Sitdikov, and Luciano Bello, with help from Caleb Johnson, Mario Motta, Andrew Eddins, Tanvi Gujarati, Stefan Wörner, Max Rossmannek, Ikko Hamamura, and Takashi Imamichi. The documentation was written by Aggie Branczyk, with help from Ieva Liepuoniute, Mario Motta and Travis Scholten.

We also thank Lev Bishop, Sarah Sheldon, and John Lapeyre for useful discussions.

## References
[1] Andrew Eddins, Mario Motta, Tanvi P. Gujarati, Sergey Bravyi, Antonio Mezzacapo, Charles Hadfield, Sarah Sheldon, *Doubling the size of quantum simulators by entanglement forging*, https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.3.010309

## License

[Apache License 2.0](LICENSE.txt)

