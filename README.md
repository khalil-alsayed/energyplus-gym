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



If you use Spyder and want it to run inside the `eplus\_test` environment:



1\. Install `spyder-kernels` in the environment:



&nbsp;   conda activate eplus\_test

&nbsp;   conda install spyder-kernels



2\. Open Spyder (from Anaconda Navigator or base environment) and go to:

&nbsp;  Tools → Preferences → Python interpreter

&nbsp;  Choose “Use the following Python interpreter” and select:



&nbsp;      <your Anaconda path>/envs/eplus\_test/python.exe



3\. Apply, restart Spyder, and open the `energyplus-gym` folder as your working directory.



