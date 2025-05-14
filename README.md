### Installation Instructions
1. Install astra-toolbox, CuPy, and ipykernel from conda-forge:
```bash
conda install -c conda-forge astra-toolbox
conda install -c conda-forge cupy
conda install -c conda-forge ipykernel
```
2. Clone the pyxalign git repo
```bash
git clone https://github.com/hruth/pyxalign.git
```
3. Install the package
```bash
cd pyxalign
pip install .
```
For an editable install, use this instead:
```bash
cd pyxalign
pip install -e .
```

To install astra-toolbox and cupy for a specific cuda-toolkit, add `cudatoolkit=[version_number]` to the end of the conda install commands. For example:
```bash
conda install -c conda-forge astra-toolbox cudatoolkit=11.8
```
This can be helpful when the conda environment is being setup on a machine that is different than the machine where the code will be run.