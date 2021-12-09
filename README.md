# Entanglement forging module
This directory contains artefacts related to the entanglement forging module.

Table of Contents:
1. <a href="#the-module">The module</a>
2. <a href="#documentation">Documentation</a>
3. <a href="#how-to-use-this-module">How to use this module</a>
4. <a href="#for-contributors">For contributors</a>
5. <a href="#contact">Contact</a>
6. <a href="#acknowledgments">Acknowledgments</a>
7. <a href="#references">References</a>
8. <a href="#license">License</a>

## The module
The module allows a user to simulate chemical and physical systems using a Variational Quantum Eigensolver (VQE) enhanced by Entanglement Forging  [[1]](./README.md#references). Entanglement Forging doubles the size of the system that can be *exactly* simulated on a fixed set of quantum bits.

The module contents can be found in the [entanglement_forging](./entanglement_forging/) directory.

## Documentation
The documentation for this module is structured around four functions:
- [Tutorials](./documentation/1-tutorials/): to teach a beginner user how the module works
- [How-to guides](./documentation/2-how_to_guides/): to show how to apply the module to solve a problem
- [A reference guide](./documentation/3-reference_guide/): technical description of the machinery of the module
- [Explanatory material](./documentation/4-explanatory_material/): discussion of concepts and further reading

The documentation can be found in the [documentation](./documentation/) directory.

Information about this documentation philosophy can be found on the [Divio website](https://documentation.divio.com).

## How to use this module

After [installing the module](./README.md#installation-instructions), we recommend that new users work their way through the examples in the [tutorials](./documentation/1-tutorials/) and the [how-to guides](./documentation/2-how_to_guides/). A brief overview of the technique can be found in [overview of Entanglement Forging](./documentation/4-explanatory_material/explanatory_material.md#overview-of-entanglement-forging) and [Entanglement Forging procedure](https://github.ibm.com/IBM-Q-Software/forging/blob/master/documentation/4-explanatory_material/explanatory_material.md#entanglement-forging-procedure).

### ⚠️ CAUTION

Before using the module for new work, users should read through the [reference guide](./documentation/3-reference_guide/) and the [explanatory material](./documentation/4-explanatory_material/), specifically the [current limitations](./documentation/4-explanatory_material/explanatory_material.md#%EF%B8%8F-current-limitations) of the module. 

### Using quantum services

If you are interested in using quantum services (i.e. using a real quantum
computer, not a simulator) you can look at the Qiskit Partners program for
partner organizations that have provider packages available for their offerings:

https://qiskit.org/documentation/partners/

### Installation instructions   
Clone the repo and install the package as follows:
```bash
git clone https://github.ibm.com/IBM-Q-Software/forging.git
cd forging
pip install -e .
```
You can now use the package. 

For detailed instructions, refer to the [installation instructions](./documentation/3-reference_guide/reference_guide.md#installation-instructions-for-users).

### How to use the installed package

Now that you have the package installed, you can follow examples within [tutorial notebooks](./documentation/tutorials/)
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
[contribution instructions](./documentation/3-reference_guide/reference_guide.md#for-contributoris-making-a-pull-request). This project adheres to Qiskit's [code of conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

We use [GitHub issues](https://github.ibm.com/IBM-Q-Software/forging/issues) for tracking requests and bugs. 

⚠️ Please note that we can't commit to implementing all feature requests, but we will review them and allocate resources when possible. 

### Installing dependencies
  
To install the dependencies needed:
```
pip install -r requirements.txt
```
For detailed instructions, refer to the [setup instructions](./documentation/3-reference_guide/reference_guide.md#for-contributors-initial-set-up-and-installing-dependencies) in the [reference guide](./documentation/3-reference_guide/).

### Running tests

See [here](./documentation/3-reference_guide/reference_guide.md#running-tests).

<!-- CONTACT -->
## Contact

Caleb Johnson - Caleb.Johnson@ibm.com

## Acknowledgments

This module is based on the theory and experiment described in [[1]](./README.md#references).

The initial code on which this module is based was written by Andrew Eddins, Mario Motta, Tanvi Gujarati, and Charles Hadfield. The module was developed by Aggie Branczyk, Iskandar Sitdikov, and Luciano Bello, with help from Caleb Johnson, Mario Motta, Andrew Eddins, Tanvi Gujarati, Stefan Wörner, Max Rossmannek, Ikko Hamamura, and Takashi Imamichi. The documentation was written by Aggie Branczyk, with help from Ieva Liepuoniute, Mario Motta and Travis Scholten.

We also thank Lev Bishop, Sarah Sheldon, and John Lapeyre for useful discussions.

## References
[1] Andrew Eddins, Mario Motta, Tanvi P. Gujarati, Sergey Bravyi, Antonio Mezzacapo, Charles Hadfield, Sarah Sheldon, *Doubling the size of quantum simulators by entanglement forging*, https://arxiv.org/abs/2104.10220

## License

[Apache License 2.0](LICENSE.txt)

