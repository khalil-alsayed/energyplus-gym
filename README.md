Quick test installation for energyplus-gym

-----------------------------------------



Follow these steps to test `energyplus-gym` on a fresh setup.



1\. Create and activate a new Conda environment



&nbsp;   conda create -n eplus\_test python=3.11

&nbsp;   conda activate eplus\_test



2\. Clone the repository



&nbsp;   cd /path/where/you/want/the/project

&nbsp;   git clone https://github.com/khalil-alsayed/energyplus-gym.git

&nbsp;   cd energyplus-gym



3\. Install the package



&nbsp;   pip install -e .

&nbsp;   # or, if you don't need editable mode:

&nbsp;   # pip install .

&nbsp;   # pip install .[examples]



4\. Verify the installation



&nbsp;   python -c "import eplus\_gym; print('energyplus-gym imported successfully')"



If this prints the message without errors, the installation works.



5\. (Optional) Run an example



&nbsp;   python examples/your\_example\_script.py



Replace `your\_example\_script.py` with the example you want to run (e.g. a Q-Transformer or DDQN demo).





Using Spyder (optional)
-----------------------

If you want to use this project in Spyder with the `eplus_test` environment:

1. Install Spyder support in the environment:

   ```bash
   conda activate eplus_test
   conda install spyder
   ```

2. Start Spyder from the same environment:

   ```bash
   spyder
   ```

3. In Spyder, set the working directory to the `energyplus-gym` folder (using the folder icon in the top right).

Now Spyder will use the Python and libraries from `eplus_test`.

Cleaning up the test environment (optional)
==========================================

If you created a temporary test setup (e.g. the `eplus_test` conda environment and a test clone of the repository) and you want to remove it, follow these steps.

1. Close Spyder and terminals
-----------------------------

- Close Spyder so it is not using the `eplus_test` environment.
- Close any Anaconda Prompt / terminal windows that are currently using that environment.

2. Delete the `eplus_test` conda environment
--------------------------------------------

1. Open **Anaconda Prompt**.
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

3. Delete the test copy of the project folder
---------------------------------------------

If you made a separate test clone of the repository, for example:

- `C:\Users\<username>\Documents\eplus_test\energyplus-gym`
- or `C:\Users\<username>\energyplus-gym-test`

you can safely delete those folders.

1. Open **File Explorer**.
2. Navigate to the parent folder (for example `C:\Users\<username>\Documents`).
3. Right-click the **test folder** (e.g. `eplus_test` or `energyplus-gym-test`) and choose **Delete**.

Be careful **not** to delete your main project folder, which might be something like:

- `C:\Users\<username>\Documents\energyplus-gym`

After these steps:
------------------

- Your original project and main conda environment remain intact.
- All temporary test environments and test clones are removed.




