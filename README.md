[![Python application test](https://github.com/MALES-project/SpeckleCn2Profiler/actions/workflows/test.yaml/badge.svg)](https://github.com/MALES-project/SpeckleCn2Profiler/actions/workflows/test.yaml)
![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/SCiarella/ee30d5a40792fc1de92e9dcf0d0e092a/raw/covbadge.json)
[![Documentation Status](https://readthedocs.org/projects/gemdat/badge/?version=latest)](https://males-project.github.io/SpeckleCn2Profiler/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/speckcn2)](https://pypi.org/project/speckcn2/)
[![PyPI](https://img.shields.io/pypi/v/speckcn2)](https://pypi.org/project/speckcn2/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11447920.svg)](https://doi.org/10.5281/zenodo.11447920)

![MALES banner](https://raw.githubusercontent.com/MALES-project/SpeckleCn2Profiler/main/src/speckcn2/assets/logo_on_white.png#gh-light-mode-only)
![MALES banner](https://raw.githubusercontent.com/MALES-project/SpeckleCn2Profiler/main/src/speckcn2/assets/logo_on_black.png#gh-dark-mode-only)

<!---
![MALES banner](https://github.com/MALES-project/SpeckleCn2Profiler/blob/main/speckcn2/assets/logo_on_white.png#gh-light-mode-only)
![MALES banner](https://github.com/MALES-project/SpeckleCn2Profiler/blob/main/speckcn2/assets/logo_on_black.png#gh-dark-mode-only)
-->

# pyrkm:
### Emergent unsupervised learning with adaptive resistor networks: The Restricted Kirchhoff Machine 

![Graphical abstract](https://github.com/MALES-project/SpeckleCn2Profiler/blob/main/src/speckcn2/assets/cn2_profile.gif?raw=true)

## Overview

## Repository Contents


## Getting Started

To get started with the project, follow these steps:

- **Prerequisites:**
  In order to correctly install `speckcn2` you need `python3.9` or higher. If you don't have it installed, you can download it from the [official website](https://www.python.org/downloads/). You will also need the header files that are required to compile Python extensions and are contained in `python3-dev`. On Ubuntu, you can install them with:
  ```bash
  apt-get install python3-dev
  ```

- **Install the package:**
   ```bash
   python -m pip install speckcn2
   ```

- **Or: Clone the repository:**
  ```bash
  git clone https://github.com/MALES-project/SpeckleCn2Profiler.git
  cd SpeckleCn2Profiler
  git submodule init
  git submodule update
  pip install .
  ```

## Usage

To use the package, you run the commands such as:

```console
python <mycode.py> <path_to_config.yml>
```

where `<mycode.py>` is the name of the script that trains/uses the `speckcn2` model and `<path_to_config.yml>` is the path to the configuration file.

[Here](https://males-project.github.io/SpeckleCn2Profiler/examples/run) you can find a typical example run and an explanation of all the main configuration parameter. In the [example submodule](https://github.com/MALES-project/examples_speckcn2/) you can find multiple examples and multiple configuration to take inspiration from.

## What can we predict?


## Contribution Guidelines

We welcome contributions to improve and expand the capabilities of this project. If you have ideas, bug fixes, or enhancements, please submit a pull request.
Check out our [Contributing Guidelines](CONTRIBUTING.md#Getting-started-with-development) to get started with development.

## Generative-AI Disclaimer

Parts of the code have been generated and/or refined using GitHub Copilot. All AI-output has been verified for correctness, accuracy and completeness, revised where needed, and approved by the author(s).

## How to cite

Please consider citing this software that is published in Zenodo under the DOI [10.5281/zenodo.11447920](https://zenodo.org/records/11447920).

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
