<div align="center">

# Attention Makes HVAC Control More Efficient

[![TestPyPI](https://img.shields.io/badge/TestPyPI-eplus--gym--khalil-blue)](https://test.pypi.org/project/eplus-gym-khalil/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![IEEE Paper (DOI)](https://img.shields.io/badge/IEEE%20Paper-DOI%3A%2010.1109%2FIECON58223.2025.11221485-blue)](https://doi.org/10.1109/IECON58223.2025.11221485)
[![TechRxiv Preprint (DOI)](https://img.shields.io/badge/TechRxiv-DOI%3A%2010.36227%2Ftechrxiv.176281127.75918518%2Fv1-orange)](https://doi.org/10.36227/techrxiv.176281127.75918518/v1)

</div>


<p align="center">
  <a href="https://iecon2025.org" target="_blank">
    <img src="https://github.com/khalil-alsayed/energyplus-gym/blob/main/logo/logo-iecon2025-trans.jpg" alt="IECON Logo" width="400"/>
  </a>
</p>

<p align="center">
  <strong>📣 Accepted at <a href="https://IECON2025.org" target="_blank">IECON 2025</a>, Madrid, Spain!</strong>
</p>


# Table of Contents
- [Features](#features)
- [Installation](#installation)
  - [Requirements](#requirements)
  - [Install package](#install-package)
  - [Verify install](#verify-install)
- [Usage](#usage)
  - [Minimal env usage](#minimal-env-usage)
  - [Training scripts](#training-scripts)
- [Project structure](#project structure(repo tree))
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)


# Features

**energyplus-gym** is a lightweight **Gymnasium-compatible** Python toolkit for running **EnergyPlus** simulations as reinforcement-learning environments for **HVAC control**. It provides a reproducible sandbox to train and evaluate deep RL agents on a realistic building case study (a **400 m² university amphitheatre**) by controlling key AHU variables such as **supply-air temperature setpoint, outdoor-air mass flow rate, and AHU ON/OFF**. 

This GitHub accompanies the paper:

📄 *Attention Makes HVAC Control More Efficient*  
- **IEEE (IECON 2025)** — DOI: [10.1109/IECON58223.2025.11221485](https://doi.org/10.1109/IECON58223.2025.11221485)  
- **TechRxiv preprint** — DOI: [10.36227/techrxiv.176281127.75918518/v1](https://doi.org/10.36227/techrxiv.176281127.75918518/v1) 

It includes:
- An EnergyPlus-based RL environment with ready-to-use building/weather assets
- Three agents implemented and benchmarked: **DDQN (MLP)**, **DDQN (Bi-LSTM)**, and the proposed **Q-Transformer** (Transformer encoder–decoder integrated into DDQN to process 24h observation sequences) 
- Scripts for **weekly adaptation** and **year-long generalisation** experiments, including cross-climate evaluation
- Tools to log and plot deployment behavior (temperature/CO₂/energy trajectories + control actions)

Overall, **energyplus-gym** aims to make sequence-aware RL for building control easier to reproduce and fairly compare, highlighting the benefit of transformer-based policies for partially observable HVAC dynamics. 


---

Installation
======================

## Requirements

<p align="center">
  <a href="https://energyplus.net" target="_blank">
    <img src="https://github.com/khalil-alsayed/energyplus-gym/blob/main/logo/energyplus.jpg" alt="energyplus Logo" width="400"/>
  </a>
</p>

To run the `eplus_gym` environments, you **must** have the EnergyPlus simulator
installed locally. The Python code in this project talks to EnergyPlus through
its official Python API, `pyenergyplus`. Without EnergyPlus, you will be able to
install the package, but any call to `env.reset()` or any simulation run will fail.

### Required EnergyPlus version

This project was developed and tested with:

    EnergyPlus 25.1

The supplied `model.idf` was generated for version 25.1 (line `Version,25.1;`).
Using the **same version** of EnergyPlus is strongly recommended to avoid IDF
compatibility issues. Newer or older versions may require upgrading the IDF
with the EnergyPlus IDFVersionUpdater tool.

### Recommended install locations (Windows)

On Windows, the recommended setup is to install EnergyPlus in the default location(s), for example:

```bash
C:\EnergyPlusV25-1-0\
```

If you install it in a different standard folder, `eplus_gym` cannot automatically detect the EnergyPlus Python API.


### Quick check

Before running any examples, you can quickly verify that the EnergyPlus Python
API is available in your environment (environment already created in #1 of Quick test installation for energyplus-gym section):

    conda activate eplus_test   # or your environment name
    python -c "from pyenergyplus.api import EnergyPlusAPI; print(EnergyPlusAPI)"

If this prints something like:

    <class 'pyenergyplus.api.EnergyPlusAPI'>

then EnergyPlus is correctly installed and its Python API is visible to your
environment. You can now start to download the project.

### Fix (if it happens): `ModuleNotFoundError: No module named 'pyenergyplus'`


If the quick check fails with:

```text
ModuleNotFoundError: No module named 'pyenergyplus'
```

it usually means EnergyPlus is installed, but its Python API folder is not on your Python path.

EnergyPlus ships the `pyenergyplus` package inside the EnergyPlus installation directory (it is not installed via `pip`).

To make it work, you must add:

- the EnergyPlus install folder to `PYTHONPATH` (so Python can import `pyenergyplus`)
- the EnergyPlus install folder to `PATH` (so Windows can find required DLLs) 

Permanent fix (recommended for Conda environments), this makes the variables load automatically every time you activate the conda env.

**Windows (Anaconda Prompt)**

1. Activate your conda environment:

```bash
conda activate eplus_test
```

2. Create activation/deactivation folders:

```bash
mkdir "%CONDA_PREFIX%\etc\conda\activate.d"
mkdir "%CONDA_PREFIX%\etc\conda\deactivate.d"
```

3. Create the activation script:

```bash
notepad "%CONDA_PREFIX%\etc\conda\activate.d\energyplus.bat"
```

Paste (edit `EPLUS_DIR` if needed):

```bash
@echo off
set "EPLUS_DIR=C:\EnergyPlusV25-1-0"

rem Save old values (so we can restore on deactivate)
set "_OLD_PYTHONPATH=%PYTHONPATH%"
set "_OLD_PATH=%PATH%"

rem Add EnergyPlus to PYTHONPATH and PATH
if exist "%EPLUS_DIR%\pyenergyplus" (
  set "PYTHONPATH=%EPLUS_DIR%;%PYTHONPATH%"
) else (
  set "PYTHONPATH=%EPLUS_DIR%\PythonAPI;%PYTHONPATH%"
)

set "PATH=%EPLUS_DIR%;%PATH%"
```

4. Create the deactivation script:

```bash
notepad "%CONDA_PREFIX%\etc\conda\deactivate.d\energyplus.bat"
```
Paste:

```bash
@echo off
set "PYTHONPATH=%_OLD_PYTHONPATH%"
set "PATH=%_OLD_PATH%"
set "_OLD_PYTHONPATH="
set "_OLD_PATH="
set "EPLUS_DIR="
```

5. Close and reopen Anaconda Prompt, activate env again, and test:

```bash
conda activate eplus_test
python -c "from pyenergyplus.api import EnergyPlusAPI; print(EnergyPlusAPI)"
```

---

## Install package

This guide shows how to quickly test `energyplus-gym` in a fresh Conda environment and, optionally, how to run it from Spyder.

### 1. Create and activate a new Conda environment

```bash
conda create -n eplus_test python=3.11
conda activate eplus_test
```

### 2. Clone the repository



You can either:

- Clone with Git (recommended, requires Git to be installed):

```bash
  cd /path/where/you/want/the/project
```
```bash
  git clone https://github.com/khalil-alsayed/energyplus-gym.git
```
```bash
  cd energyplus-gym
```

- Or download as ZIP (no Git needed):

  1. Go to the GitHub page of the project.
  2. Click "Code" → "Download ZIP".
  3. Unzip the archive.
  4. Open a terminal / Anaconda Prompt and move into the project folder:
     ```bash
     cd /path/to/unzipped/energyplus-gym
     ```
### 3. Install the package

### User installation (recommended)

If (and only if) you previously installed `eplus_gym`, uninstall it first to avoid importing an old `site-packages`:

```bash
pip uninstall -y eplus-gym eplus_gym
```
 
Now install dependencies:

```bash
pip install .
```

and, to also install example dependencies:

```bash
pip install .[examples]
```
### OR Developer / contributor installation (editable)

If you previously installed an older version, remove it (optional but recommended)

```bash
pip uninstall -y eplus-gym eplus_gym
```
Now install in editable mode

```bash
pip install -e .
```

also install example dependencies:

```bash
pip install -e ".[examples]"
```


### Fix (if it happens): `error: subprocess-exited-with-error` & `error: metadata-generation-failed`

On Windows this usually happens when pip cannot find a prebuilt NumPy wheel that matches your Python, so it falls back to compiling (which needs Visual C++ Build Tools).

1. Install Visual Studio Build Tools 2022

- Select: Desktop development with C++
- Make sure these are included:
    - MSVC v143 build tools
    - Windows 10/11 SDK

2. Close and reopen Anaconda Prompt, then:
   
```bash
conda activate eplus_test
python -m pip install -U pip setuptools wheel
pip install .
pip install .[examples]
```
 Or for Developer / contributor installation

```bash
conda activate eplus_test
python -m pip install -U pip setuptools wheel
pip install -e .
pip install -e ".[examples]"
```

### 4. Verify the installation


Run a quick import test:

```bash
python -c "import eplus_gym; print('energyplus-gym imported successfully')"
```

If this prints the message without errors, the installation works.

# Usage

## Using Spyder 

<p align="center">
  <a href="https://www.spyder-ide.org/" target="_blank">
    <img src="https://github.com/khalil-alsayed/energyplus-gym/blob/main/logo/spyder.png" alt="spyder Logo" width="400"/>
  </a>
</p>

To use this project in Spyder with the `eplus_test` environment:

### 1. Install Spyder inside the environment (once):

```bash
   conda activate eplus_test
   conda install spyder
```

### 2. Start Spyder from the same environment:

```bash
   spyder
```

### 3. Configure the working directory to the Q-Transformer example folder:

   This ensures that scripts such as `main.py` can import `dqn_agent.py` and other helper files without additional path tweaks.

   - In Spyder, open:
     Tools -> Preferences -> Working directory
   - Under "Startup" -> "The following directory", set the path to:

     path_where_you_cloned_the_project\energyplus-gym\src\eplus_gym\agents

     For example:

     C:\Users\khali\Documents\energyplus-gym\src\eplus_gym\agents

   - (Optional but recommended) Under "New consoles", also select
     "The following directory" and use the same path.

   - Click Apply, then OK.

### 4. Restart the IPython kernel / console in Spyder:

   - Either click "Restart kernel" in the IPython console, or
   - Close the current console and open a new one.

After this, Spyder will:

- Use the Python and libraries from the `eplus_test` environment, and
- Start in energyplus-gym\src\eplus_gym\agents, so all Python scripts and imports in that folder will work automatically.


# Project structure (repo tree)

```text
.
├── README.md
├── pyproject.toml
├── MANIFEST.in
├── LICENSE-PAPER.md
├── examples/
│   └── quickstart.py
├── scripts/
│   ├── train_adaptation_*.py
│   ├── train_generalisation_*.py
│   ├── eval_adaptation_*.py
│   └── eval_generalisation_*.py
├── src/
│   └── eplus_gym/
│       ├── __init__.py
│       ├── envs/
│       │   ├── env.py
│       │   ├── energyplus.py
│       │   ├── utils.py
│       │   └── assets/
│       │       ├── model.idf
│       │       ├── *.epw
│       │       └── normalization/
│       │           └── *.csv
│       └── agents/
│           ├── ddqn_mlp/
│           ├── ddqn_bilstm/
│           └── q_transformer/
└── tests/
    └── test_smoke.py

> Note: `runs/`, `eplus_outputs/`, `dist/`, `__pycache__/`, and `*.egg-info/` are generated during training/runs/build and are ignored by Git.


```
#6. Cleaning up the test environment (optional)
----------------------------------------------

If you created a temporary test setup (for example, the `eplus_test` Conda environment and a test clone of the repository) and you want to remove it, follow these steps.

6.1 Close Spyder and terminals
------------------------------

- Close Spyder so it is not using the `eplus_test` environment.
- Close any Anaconda Prompt / terminal windows that are currently using that environment.

6.2 Delete the `eplus_test` Conda environment
---------------------------------------------

1. Open Anaconda Prompt.
2. List your environments (optional, just to see their names):

 ```bash
   conda env list
 ```

3. Make sure you are not inside `eplus_test`:

```bash
   conda deactivate
```

4. Remove the test environment:

```bash
   conda remove --name eplus_test --all
```

5. Verify that it is gone:

```bash
   conda env list
```

If you created additional test environments (for example `test_eplusgym`), you can remove them in the same way:

```bash
conda remove --name test_eplusgym --all
```

6.3 Delete the test copy of the project folder
----------------------------------------------

If you made a separate test clone of the repository, for example:

- C:\Users\<username>\Documents\eplus_test\energyplus-gym
- or C:\Users\<username>\energyplus-gym-test

you can safely delete those folders.

1. Open File Explorer.
2. Navigate to the parent folder (for example C:\Users\<username>\Documents).
3. Right-click the test folder (e.g. eplus_test or energyplus-gym-test) and choose Delete.

Be careful not to delete your main project folder, which might be something like:

- C:\Users\<username>\Documents\energyplus-gym

After these steps:
------------------

- Your original project and main Conda environment remain intact.
- All temporary test environments and test clones are removed.


## License & Attribution

This project is licensed under the **Khalil Al Sayed Community Research License**. 
See the `LICENSE` file for full terms.

### Credits
This work incorporates and modifies code originally developed by **Antoine Galataud** (2022). 
The original components remain under the MIT License, and the full notice can be found 
in the `NOTICE` file included in this repository.
