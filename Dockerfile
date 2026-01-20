# Multi-stage Dockerfile for PyRKM
# Stage 1: Base image with Python and system dependencies
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Stage 2: Development image with all dev dependencies
FROM base AS development

# Copy requirements first for better caching
COPY requirements.txt requirements_full.txt ./
COPY pyproject.toml MANIFEST.in ./

# Install PyTorch with CUDA support
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install all dependencies including development tools
RUN pip install -e ".[develop,docs]"

# Install additional development tools
RUN pip install \
    ipython \
    jupyter \
    jupyterlab \
    wandb \
    tensorboard \
    pre-commit

# Copy the entire project
COPY . .

# Install the package in development mode
RUN pip install -e .

# Setup pre-commit hooks
RUN git init || true && pre-commit install || true

# Expose ports for Jupyter and TensorBoard
EXPOSE 8888 6006

# Default command for development
CMD ["bash"]

# Stage 3: Production image with minimal dependencies
FROM base AS production

# Copy requirements
COPY requirements.txt ./
COPY pyproject.toml MANIFEST.in ./

# Install PyTorch with CUDA support
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install only runtime dependencies
RUN pip install -e .

# Copy only source code
COPY src/ ./src/
COPY README.md LICENSE ./

# Install the package
RUN pip install .

# Create non-root user for security
RUN useradd -m -u 1000 pyrkm && \
    chown -R pyrkm:pyrkm /app

USER pyrkm

# Default command
CMD ["python"]

# Stage 4: Jupyter notebook server
FROM development AS jupyter

# Expose Jupyter port
EXPOSE 8888

# Configure Jupyter
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Stage 5: CPU-only image for environments without GPU
FROM python:3.10-slim AS cpu-only

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt pyproject.toml MANIFEST.in ./

# Install PyTorch CPU version
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy and install the package
COPY . .
RUN pip install -e .

CMD ["python"]
