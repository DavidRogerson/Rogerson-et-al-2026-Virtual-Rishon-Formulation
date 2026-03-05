# Source Code: "Simulating Lattice Gauge Theories with Virtual Rishons"

This repository contains the source code accompanying the publication: 
**"Simulating Lattice Gauge Theories with Virtual Rishons"**, David Rogerson et al., 2026.

## Structure
The repository is organized into four main components:

### 1. Environment Setup (`environment.yml`)
Contains the configuration for the `micromamba`/`anaconda` environment required to run the code, ensuring all dependencies (TeNPy, Signac, etc.) are correctly versioned.

### 2. Core Implementation (`src/Rogerson_et_al_2026_Virtual_Rishon_Formulation`)
A reference implementation of the Virtual Rishon formulation based on the [TeNPy library](https://tenpy.readthedocs.io/).
* **Models:** * The Schwinger model implementation: `models.massive_schwinger_model_qubit_encoding.py`
    * The QED$_3$ implementation: `models.QED3_qubit_encoding.py`
* **Lattice Geometry:** Defined in `models.lattice.LatticeGaugeTheoryLattice` and `networks.site`.
* **MPO Compression:** A specialized implementation based on [Parker2020](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.102.035147) can be found in `networks.mpo.MPOCompress`. This is used in the QED$_3$ model.
* **Configuration:** Includes boilerplate code for verbose parameter definitions via `Config` classes (e.g., `CLASSNAMEConfig`).

### 3. Data Management and Workflow
* **Data Storage:** The simulation data is structured using [signac](https://github.com/glotzerlab/signac). Due to its size, the raw data is hosted on [Zenodo](https://zenodo.org/records/18864580) but is meant to be extracted into the workspace folder.
* **Automation:** Data can be generated systematically and reproducibly using the scripts located in `scripts/` via the [row](https://github.com/glotzerlab/row) framework.
* **Minimal Example:** A "how-to" guide for producing new data is provided in `notebooks/Minimal_example.ipynb`.

### 4. Analysis and Figures
Notebooks used to generate the figures and plots found in the main publication are stored in `notebooks/analysis/`, but requires the dataset mentioned above.
