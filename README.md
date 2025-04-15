### Installation Instructions
1. Install the astra-toolbox from conda-forge
```bash
conda install -c conda-forge astra-toolbox
```
2. Install CuPy from conda-forge
```bash
conda install -c conda-forge cupy
```
3. Clone the llama git repo
```bash
git clone https://github.com/hruth/llama.git
```
4. Install the package
```bash
pip install llama
```
For an editable install, use this instead:
```bash
pip install -e llama
```

To install astra-toolbox and cupy for a specific cuda-toolkit, add `cudatoolkit=[version_number]` to the end of the conda install commands. For example:
```bash
conda install -c conda-forge astra-toolbox cudatoolkit=11.8
```
This can be helpful when the conda environment is being setup on a machine that is different than the machine where the code will be run.