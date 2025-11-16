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
   conda install spyder-kernels
   ```

2. Start Spyder from the same environment:

   ```bash
   spyder
   ```

3. In Spyder, set the working directory to the `energyplus-gym` folder (using the folder icon in the top right).

Now Spyder will use the Python and libraries from `eplus_test`.




