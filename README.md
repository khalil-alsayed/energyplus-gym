Quick test installation for `energyplus-gym`
===========================================

This guide shows how to quickly test `energyplus-gym` in a fresh Conda environment and, optionally, how to run it from Spyder.

1. Create and activate a new Conda environment
----------------------------------------------

```bash
conda create -n eplus_test python=3.11
conda activate eplus_test
```

2. Clone the repository
-----------------------

Choose where you want to keep the project and clone it there:

```bash
cd /path/where/you/want/the/project
```

git clone https://github.com/khalil-alsayed/energyplus-gym.git

```bash
cd energyplus-gym
```

3. Install the package
----------------------

Editable install (recommended while developing):

```bash
pip install -e .
```

If you don’t need editable mode:

```bash
pip install .
```

# or, to also install example dependencies:

```bash
pip install .[examples]
```

4. Verify the installation
--------------------------

Run a quick import test:

```bash
python -c "import eplus_gym; print('energyplus-gym imported successfully')"
```

If this prints the message without errors, the installation works.

5. Using Spyder 
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

     path_where_you_cloned_the_project\energyplus-gym\examples\Q-transformer

     For example:

     C:\Users\khali\Documents\energyplus-gym\examples\Q-transformer

   - (Optional but recommended) Under "New consoles", also select
     "The following directory" and use the same path.

   - Click Apply, then OK.

4. Restart the IPython kernel / console in Spyder:

   - Either click "Restart kernel" in the IPython console, or
   - Close the current console and open a new one.

After this, Spyder will:

- Use the Python and libraries from the `eplus_test` environment, and
- Start in energyplus-gym\examples\Q-transformer, so all Python scripts and imports in that folder (e.g. `from dqn_agent import DQNAgent`) will work automatically.

6. Cleaning up the test environment (optional)
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
