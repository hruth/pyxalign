# Use NVIDIA's official CUDA base image with Ubuntu 22.04
# This already includes the CUDA toolkit and necessary drivers
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Set environment variables
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda

# Set the working directory
WORKDIR /app
# copy pyxalign into app
COPY . /app/pyxalign

# Install system dependencies and Python 3.13
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.13 \
    python3.13-venv \
    python3.13-dev \
    python3-pip \
    build-essential \
    curl \
    wget \
    git \
    libopenblas-dev \
    libffi-dev \
    libssl-dev \
    libqt5gui5 \
    python3.13-tk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment using Python 3.13
RUN python3.13 -m venv $VIRTUAL_ENV

# Upgrade pip, setuptools, and wheel inside the virtual environment
RUN pip install --upgrade pip setuptools wheel

# Install CuPy with CUDA 12.x support
# Change 'cupy-cuda12x' to match your CUDA version:
#   cupy-cuda11x  -> CUDA 11.x
#   cupy-cuda12x  -> CUDA 12.x
# RUN pip install cupy-cuda12x && \
#     pip install h5py && \
#     pip install matplotlib && \
#     pip install numpy && \
#     pip install pandas && \
#     pip install scipy && \
#     pip install scikit-image && \
#     pip install statsmodels && \
#     pip install tqdm && \
#     pip install h5py && \
#     pip install matplotlib && \
#     pip install ipywidgets && \
#     pip install plotly && \
#     pip install ipympl && \
#     pip install tifffile && \
#     pip install PyQt5 && \
#     pip install mda-xdrlib && \
#     pip install PyYAML && \
#     pip install astra-toolbox && \
#     pip install pyqtgraph && \
#     pip install jupyter

RUN cd /app/pyxalign && \
    pip install -e .[cupy12]

# install dependencies manually or update pyproject.toml? 

CMD ["/bin/bash"]