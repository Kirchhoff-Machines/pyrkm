# Docker Guide for PyRKM

This project provides a simple Docker environment with GPU support for running the PyRKM code and example notebooks.

## Prerequisites

1.  **Docker Engine**: Install Docker Engine (version 19.03 or later).
2.  **NVIDIA Container Toolkit**: Since this project uses GPU acceleration, you must install the `nvidia-container-toolkit`.
    *   [Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
    *   This allows Docker containers to see and use your NVIDIA GPU.
3.  **Docker Compose V2**: We recommend using the modern `docker compose` command (integrated into the Docker CLI) rather than the standalone legacy `docker-compose` (Python script).

## Detailed Setup Instructions

### 1. Build and Start the Environment

Open your terminal in the root of the repository and run:

```bash
docker compose up --build
```

**Note:** If you get a "command not found" error for `docker compose`, you might be using an older version of Docker. You can try `docker-compose up --build` instead, but you may run into compatibility issues with newer Python versions (like the `ModuleNotFoundError: No module named 'distutils'` error).

**What this does:**
*   Downloads the base NVIDIA CUDA image.
*   Installs Python 3.10 and all necessary system libraries.
*   Installs PyTorch (GPU version) and all project dependencies listed in `requirements.txt`.
*   Starts a Jupyter Lab server accessible from your browser.
*   Mounts your current directory inside the container, so changes to your code locally are reflected instantly.

### 2. Access Jupyter Lab

Once the container starts, you will see logs flowing in your terminal. Look for a line that looks like this:

```text
pykr-gpu  |     http://127.0.0.1:8888/lab?token=a1b2c3d4e5f6...
```

**Copy and paste this full URL into your web browser.**

### 3. Running the Example

1.  In the Jupyter Lab interface (left sidebar), navigate to the `docs/examples/` folder.
2.  Double-click `first_example.ipynb` to open it.
3.  **Verify GPU Access**: The first few cells of the notebook check for a GPU. Run them (Shift+Enter). You should see output indicating a CUDA device is available (e.g., `cuda`, or a device name).
4.  Run through the rest of the notebook to train the RKM model.

### 4. Troubleshooting

**Error: `ModuleNotFoundError: No module named 'distutils'`**
This happens if you use the old `docker-compose` (v1) command with Python 3.12, which has removed `distutils`.
*   **Solution**: Use `docker compose` (with a space) instead of `docker-compose` (with a hyphen). This uses the newer Docker CLI plugin which relies on Go, not Python, and avoids this error.

**Permissions Errors**
If you have issues editing files from inside Docker (or vice versa) on Linux, it may be due to user permissions.
*   The simplified Docker setup runs as root inside the container to avoid complexity. Just be aware that files created inside the container *might* belong to root.
*   To fix file ownership on your host machine after stopping Docker, you can run: `sudo chown -R $USER:$USER .`

### 5. Stopping the Environment

To stop the container, verify you are in the terminal where it is running and press `Ctrl+C`.
Alternatively, open a new terminal in the same directory and run:

```bash
docker compose down
```
