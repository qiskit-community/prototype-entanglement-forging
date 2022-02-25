<!-- SHIELDS -->
<div align="left">

  ![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS-informational)
  [![Python](https://img.shields.io/badge/Python-3.7%20%7C%203.8%20%7C%203.9-informational)](https://www.python.org/)
  [![Qiskit](https://img.shields.io/badge/Qiskit-%E2%89%A5%200.34.1-6133BD)](https://github.com/Qiskit/qiskit)
  [![Qiskit-Nature](https://img.shields.io/badge/Qiskit--Nature-%E2%89%A5%200.3.0-6133BD)](https://github.com/Qiskit/qiskit-nature)
<br />
  [![License](https://img.shields.io/github/license/IBM-Quantum-Prototypes/entanglement-forging?color=black&label=License)](https://github.com/IBM-Quantum-Prototypes/entanglement-forging/blob/main/LICENSE.txt)
  [![Tests](https://github.com/IBM-Quantum-Prototypes/entanglement-forging/actions/workflows/makefile.yml/badge.svg)](https://github.com/IBM-Quantum-Prototypes/entanglement-forging/actions/workflows/makefile.yml)

</div>

----------------------------------------------------------------------------------------------------

<!-- PROJECT LOGO -->
<p align="center">
  <h2 align="center">Entanglement Forging</h2>
</p>
<p align="center">
  <a href="entanglement-forging">
    <img src="docs/images/ef_image.png" alt="Logo" width="600">
  </a>
</p>
<p align="center">
  <a href="https://mybinder.org/v2/gh/IBM-Quantum-Prototypes/entanglement-forging/HEAD?labpath=docs%2F1-tutorials%2Ftutorial_2_H2O_molecule_statevector_simulator.ipynb">
    <img src="https://img.shields.io/badge/launch-demo-579ACA.svg?style=for-the-badge&logo=data:image/png;base64,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" alt="Launch Demo">
  </a>
</p>


----------------------------------------------------------------------------------------------------

<!-- TABLE OF CONTENTS -->
### Table of Contents
* [Installation](docs/2-reference_guide/reference_guide.md#installation-instructions)
* [Tutorials](docs/1-tutorials/)
* [Background](docs/3-explanatory_material/)
* [How to Give Feedback](#how-to-give-feedback)
* [Contribution Guidelines](#contribution-guidelines)
* [Acknowledgements](#acknowledgements)
* [References](#references)
* [License](#license)
----------------------------------------------------------------------------------------------------


<!-- ABOUT THIS PROJECT -->
### About This Project
This module allows a user to simulate chemical and physical systems using a Variational Quantum Eigensolver (VQE) enhanced by Entanglement Forging [[1]](#references). Entanglement Forging doubles the size of the system that can be *exactly* simulated on a fixed set of quantum bits.

Before using the module for new work, users should read through the [reference guide](./docs/2-reference_guide/reference_guide.md) and the [explanatory material](docs/3-explanatory_material/explanatory_material.md), specifically the [current limitations](docs/3-explanatory_material/explanatory_material.md#%EF%B8%8F-current-limitations) of the module.


----------------------------------------------------------------------------------------------------

<!-- HOW TO GIVE FEEDBACK -->
### How to Give Feedback
We encourage your feedback! You can share your thoughts with us by:
- [Opening an issue](https://github.com/IBM-Quantum-Prototypes/entanglement-forging/issues) in the repository
- [Starting a conversation on GitHub Discussions](https://github.com/IBM-Quantum-Prototypes/entanglement-forging/discussions)
- Filling out our [survey](https://airtable.com/shrFxJXYzjxf5tFvx)


----------------------------------------------------------------------------------------------------

<!-- CONTRIBUTION GUIDELINES -->
### Contribution Guidelines
For information on how to contribute to this project, please take a look at our [CONTRIBUTING.MD](CONTRIBUTING.md) and the [Contribution Guide](https://github.com/IBM-Quantum-Prototypes/docs/2-reference-guide/reference_guide.md#contribution-guide) section of the reference guide.


----------------------------------------------------------------------------------------------------

<!-- ACKNOWLEDGEMENTS -->
### Acknowledgements
This module is based on the theory and experiment described in [[1]](#references).

The initial code on which this module is based was written by Andrew Eddins, Mario Motta, Tanvi Gujarati, and Charles Hadfield. The module was developed by Aggie Branczyk, Iskandar Sitdikov, and Luciano Bello, with help from Caleb Johnson, Mario Motta, Andrew Eddins, Tanvi Gujarati, Stefan Wörner, Max Rossmannek, Ikko Hamamura, and Takashi Imamichi. The documentation was written by Aggie Branczyk, with help from Ieva Liepuoniute, Mario Motta and Travis Scholten.

We also thank Lev Bishop, Sarah Sheldon, and John Lapeyre for useful discussions.


----------------------------------------------------------------------------------------------------

<!-- REFERENCES -->
### References
[1] Andrew Eddins, Mario Motta, Tanvi P. Gujarati, Sergey Bravyi, Antonio Mezzacapo, Charles Hadfield, Sarah Sheldon, *Doubling the size of quantum simulators by entanglement forging*, https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.3.010309


----------------------------------------------------------------------------------------------------

<!-- LICENSE -->
### License
[Apache License 2.0](LICENSE.txt)
