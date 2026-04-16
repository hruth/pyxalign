### Installation Instructions: Conda
1. Create a conda environment with python 3.13*
```bash
conda create -n pyxalign python=3.13
```
then activate the environment
```bash
conda activate pyxalign
```
2. Install astra-toolbox and CuPy from conda-forge:
```bash
conda install -c conda-forge astra-toolbox
conda install -c conda-forge cupy
```
Installing these libraries from conda-forge instead of pip ensures that required CUDA toolkit is installed.
3. Clone the pyxalign git repo
```bash
git clone https://github.com/AdvancedPhotonSource/pyxalign.git
```
4. Install the package
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

### Installation Instructions: Container

1. Clone the pyxalign git repo
```bash
git clone https://github.com/AdvancedPhotonSource/pyxalign.git
cd pyxalign
```
2. Build the container image
```bash
podman build -t pyxalign:latest .
```
3. Run the container
```bash
podman run \
    -it --rm --env DISPLAY --security-opt label=type:container_runtime_t \
    --network host -v="$HOME/.Xauthority:/root/.Xauthority:rw" \
    --device nvidia.com/gpu=all \
    pyxalign:latest bash
```
Any directories you need access to inside the container must be explicitly mounted using the `-v` flag:
```bash
-v /path/on/host:/path/in/container
```
For example, to mount your data directory:
```bash
podman run \
    -it --rm --env DISPLAY --security-opt label=type:container_runtime_t \
    --network host -v="$HOME/.Xauthority:/root/.Xauthority:rw" \
    -v /data/my_experiment:/data/my_experiment \
    --device nvidia.com/gpu=all \
    pyxalign:latest bash
```