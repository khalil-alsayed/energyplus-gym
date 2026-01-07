<div align="center">

# Attention Makes HVAC Control More Efficient

[![TestPyPI](https://img.shields.io/badge/TestPyPI-eplus--gym--khalil-blue)](https://test.pypi.org/project/eplus-gym-khalil/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![IEEE Paper (DOI)](https://img.shields.io/badge/IEEE%20Paper-DOI%3A%2010.1109%2FIECON58223.2025.11221485-blue)](https://doi.org/10.1109/IECON58223.2025.11221485)
[![TechRxiv Preprint (DOI)](https://img.shields.io/badge/TechRxiv-DOI%3A%2010.36227%2Ftechrxiv.176281127.75918518%2Fv1-orange)](https://doi.org/10.36227/techrxiv.176281127.75918518/v1)

</div>


EnergyPlus requirement
======================

To run the `eplus_gym` environments, you **must** have the EnergyPlus simulator
installed locally. The Python code in this project talks to EnergyPlus through
its official Python API, `pyenergyplus`. Without EnergyPlus, you will be able to
install the package, but any call to `env.reset()` or any simulation run will fail.

Required EnergyPlus version
---------------------------

This project was developed and tested with:

    EnergyPlus 25.1

The supplied `model.idf` was generated for version 25.1 (line `Version,25.1;`).
Using the **same version** of EnergyPlus is strongly recommended to avoid IDF
compatibility issues. Newer or older versions may require upgrading the IDF
with the EnergyPlus IDFVersionUpdater tool.

Recommended install locations (Windows)
---------------------------------------

On Windows, the simplest setup is to install EnergyPlus in one of the default
locations, for example:

```bash
    C:\EnergyPlusV25-1-0\
    C:\Program Files\EnergyPlusV25-1-0\
```

If you use one of these standard folders, `eplus_gym` can usually detect the
EnergyPlus Python API automatically.

Quick check
-----------

Before running any examples, you can quickly verify that the EnergyPlus Python
API is available in your environment:

    conda activate eplus_test   # or your environment name
    python -c "from pyenergyplus.api import EnergyPlusAPI; print(EnergyPlusAPI)"

If this prints something like:

    <class 'pyenergyplus.api.EnergyPlusAPI'>

then EnergyPlus is correctly installed and its Python API is visible to your
environment. You can now start to download the project.


Quick test installation for `energyplus-gym`
===========================================

This guide shows how to quickly test `energyplus-gym` in a fresh Conda environment and, optionally, how to run it from Spyder.

#1. Create and activate a new Conda environment
----------------------------------------------

```bash
conda create -n eplus_test python=3.11
conda activate eplus_test
```

#2. Clone the repository
-----------------------


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
#3. Install the package
----------------------
## User installation (recommended)

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
## OR Developer / contributor installation (editable)

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

#4. Verify the installation
--------------------------

Run a quick import test:

```bash
python -c "import eplus_gym; print('energyplus-gym imported successfully')"
```

If this prints the message without errors, the installation works.

#5. Using Spyder 
--------------------------

To use this project in Spyder with the `eplus_test` environment:

1. Install Spyder inside the environment (once):

```bash
   conda activate eplus_test
   conda install spyder
```

2. Start Spyder from the same environment:

```bash
   spyder
```

3. Configure the working directory to the Q-Transformer example folder:

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

4. Restart the IPython kernel / console in Spyder:

   - Either click "Restart kernel" in the IPython console, or
   - Close the current console and open a new one.

After this, Spyder will:

- Use the Python and libraries from the `eplus_test` environment, and
- Start in energyplus-gym\src\eplus_gym\agents, so all Python scripts and imports in that folder will work automatically.

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


## Licensing
- Code: MIT (see `LICENSE`)
- Paper/preprint & supplementary research materials: Khalil Al Sayed Community Research License (see `LICENSE-PAPER.md`)

