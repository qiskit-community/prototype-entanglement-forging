<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="quantum-kernel-training">
    <h2 align="center">Entanglement Forging Toolkit</h2>
    <img src="docs/images/ef_image.png" alt="Logo" width="600">
  <p align="center">
   <a href="docs/1-tutorials/tutorial_1_H2_molecule_statevector_simulator.ipynb">View Demo</a>
  </p>
</p>


<!-- TABLE OF CONTENTS -->
### Table of Contents
* [Installation](docs/2-reference_guide/reference_guide.md#installation-instructions)
* [Tutorials](docs/1-tutorials/)
* [Background](docs/3-explanatory_material/)
* [Using Quantum Services](#using-quantum-services)
* [How to Give Feedback](#how-to-give-feedback)
* [Contribution Guidelines](#contribution-guidelines)
* [Acknowledgements](#acknowledgements)
* [References](#references)
* [License](#license)


----------------------------------------------------------------------------------------------------

<!-- ABOUT THIS PROJECT -->
### About This Project
This module allows a user to simulate chemical and physical systems usin a Variational Quantum Eigensolver (VQE) enhanced by Entanglement Forging [[1]](#references). Entanglement Forging doubles the size of the system that can be *exactly* simulated on a fixed set of quantum bits.

### ⚠️ CAUTION

Before using the module for new work, users should read through the [reference guide](./docs/2-reference_guide/reference_guide.md) and the [explanatory material](docs/3-explanatory_material/explanatory_material.md), specifically the [current limitations](docs/3-explanatory_material/explanatory_material.md#current-limitations) of the module.

***Documentation***

The documentation can be found in the [docs](docs/) directory.

The documentation for this module is structured around three functions:
- [Tutorials](docs/1-tutorials): Teach a beginner how the module works
- [Reference Guide](docs/2-reference_guide/reference_guide.md): Technical description of the software
- [Explanatory Material](docs/3-explanatory_material/explanatory_material.md): Discussion of concepts and further reading

----------------------------------------------------------------------------------------------------

<!-- USING QUANTUM SERVICES -->
### Using Quantum Services
If you are interested in using quantum services (i.e. using a real quantum computer, not a simulator) you can look at the Qiskit Partners program for partner organizations that have provider packages available for their offerings:

https://qiskit.org/documentation/partners/


----------------------------------------------------------------------------------------------------

<!-- HOW TO GIVE FEEDBACK -->
### How to Give Feedback
We encourage your feedback! You can share your thoughts with us by:
- [Opening an issue](https://github.com/IBM-Quantum-Prototypes/quantum-kernel-training/issues) in the repository
- [Starting a conversation on GitHub Discussions](https://github.com/IBM-Quantum-Prototypes/quantum-kernel-training/discussions)
- Filling out our [survey](https://airtable.com/shrFxJXYzjxf5tFvx)


----------------------------------------------------------------------------------------------------

<!-- CONTRIBUTION GUIDELINES -->
### Contribution Guidelines
For information on how to contribute to this project, please take a look at our [contribution guidelines](CONTRIBUTING.md).


----------------------------------------------------------------------------------------------------

<!-- ACKNOWLEDGEMENTS -->
### Acknowledgements
This module is based on the theory and experiment described in [[1]](#references).


----------------------------------------------------------------------------------------------------

<!-- REFERENCES -->
### References
[1] Andrew Eddins, Mario Motta, Tanvi P. Gujarati, Sergey Bravyi, Antonio Mezzacapo, Charles Hadfield, Sarah Sheldon, *Doubling the size of quantum simulators by entanglement forging*, https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.3.010309


----------------------------------------------------------------------------------------------------

<!-- LICENSE -->
### License
[Apache License 2.0](LICENSE.txt)
