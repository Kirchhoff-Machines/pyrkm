[![Python application test](https://github.com/Kirchhoff-Machines/pyrkm/actions/workflows/test.yaml/badge.svg)](https://github.com/Kirchhoff-Machines/pyrkm/actions/workflows/test.yaml)
![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/SCiarella/07cc3e145e56231a59fde8156485519b/raw/coverage_pyrkm.json)
[![Documentation Status](https://github.com/Kirchhoff-Machines/pyrkm/actions/workflows/docs.yaml/badge.svg)](https://kirchhoff-machines.github.io/pyrkm/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyrkm)](https://pypi.org/project/pyrkm/)
[![PyPI](https://img.shields.io/pypi/v/pyrkm)](https://pypi.org/project/pyrkm/)
[![DOI](https://zenodo.org/badge/928211837.svg)](https://doi.org/10.5281/zenodo.14865380)


![pyrkm banner](https://raw.githubusercontent.com/Kirchhoff-Machines/pyrkm/main/src/pyrkm/assets/logo-black.png#gh-light-mode-only)
![pyrkm banner](https://raw.githubusercontent.com/Kirchhoff-Machines/pyrkm/main/src/pyrkm/assets/logo-white.png#gh-dark-mode-only)


## What is a Restricted Kirchhoff Machine?
You may be familiar with Restricted Boltzmann Machines (RBMs) [[1](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.35.1792)]-[[2](https://www.science.org/doi/10.1126/science.1127647)], which are a type of generative neural network that can learn a probability distribution over its input data. The Restricted Kirchhoff Machine (RKM) is a realization of a RBM using resistor networks, and Kirchhoff's laws of electrical circuits. In this repository, we provide a Python package to virtually simulate the training and evaluation of RKMs.

For more information about the capabilities of the RKM, see the original paper by [Link to arXiv paper](https://arxiv.org/abs/2509.15842).

## Repository Contents
In this repository you will find the following:

- `src/pyrkm/`: The main package code. You can use this code to train and evaluate RKMs. For more information, see the [documentation](https://kirchhoff-machines.github.io/pyrkm/). For a quick start, see the [Usage](#Usage) section below.
- `energy_consumption`: A series of scripts to evaluate the energy consumption of the RKM and compare it to the estimated cost of a RBM. They are used to generate the results in the [paper](https://arxiv.org/abs/2509.15842).

## Getting Started

To get started with the project, follow these steps:

- **Prerequisites:**
  In order to correctly install `pyrkm` you need `python3.9` or higher. If you don't have it installed, you can download it from the [official website](https://www.python.org/downloads/).

- **Install the package:**
   ```bash
   python -m pip install pyrkm
   ```

- **Or: Clone the repository:**
  ```bash
  git clone https://github.com/Kirchhoff-Machines/pyrkm.git
  cd pyrkm
  git submodule init
  git submodule update
  pip install .
  ```

## Usage

To learn how to use the package, follow the [official documentation](https://kirchhoff-machines.github.io/pyrkm/) and in particular [this tutorial](https://kirchhoff-machines.github.io/pyrkm/examples/first_example/).

## Contribution Guidelines

We welcome contributions to improve and expand the capabilities of this project. If you have ideas, bug fixes, or enhancements, please submit a pull request.
Check out our [Contributing Guidelines](CONTRIBUTING.md#Getting-started-with-development) to get started with development.

## Generative-AI Disclaimer

Parts of the code have been generated and/or refined using GitHub Copilot. All AI-output has been verified for correctness, accuracy and completeness, revised where needed, and approved by the author(s).

## How to cite

Please consider citing this software that is published in Zenodo under the DOI [10.5281/zenodo.14865380](https://doi.org/10.5281/zenodo.14865380).

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
