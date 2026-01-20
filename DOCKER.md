# Docker Guide for PyRKM

This guide explains how to use Docker to create reproducible development and production environments for PyRKM.

## Prerequisites

### For GPU Support

- Docker Engine 19.03 or later
- NVIDIA Docker runtime (`nvidia-docker2`)
- NVIDIA GPU with CUDA support
- NVIDIA drivers installed on host

To install NVIDIA Docker:

```bash
# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### For CPU-Only

- Docker Engine 19.03 or later
- Docker Compose

## Quick Start

### Using Make (Recommended)

The project includes a Makefile for easy Docker management:

```bash
# View all available commands
make help

# Build and start development environment (GPU)
make build
make dev

# Build and start development environment (CPU-only)
make build-cpu
make dev-cpu

# Start Jupyter Lab
make jupyter          # With GPU
make jupyter-cpu      # CPU-only

# Run tests
make test

# Start TensorBoard
make tensorboard
```

### Using Docker Compose Directly

#### GPU-Enabled Environment

```bash
# Build all images
docker-compose build

# Start development container
docker-compose up -d pyrkm-dev
docker-compose exec pyrkm-dev bash

# Start Jupyter Lab
docker-compose up -d pyrkm-jupyter
# Access at http://localhost:8888

# Start TensorBoard
docker-compose up -d tensorboard
# Access at http://localhost:6006
```

#### CPU-Only Environment

```bash
# Build CPU images
docker-compose -f docker-compose.cpu.yml build

# Start development container
docker-compose -f docker-compose.cpu.yml up -d pyrkm-dev-cpu
docker-compose -f docker-compose.cpu.yml exec pyrkm-dev-cpu bash

# Start Jupyter Lab
docker-compose -f docker-compose.cpu.yml up -d pyrkm-jupyter-cpu
```

## Docker Images

The project provides several Docker images:

### 1. Development Image (`pyrkm:dev`)

- Full development environment with all dependencies
- Pre-commit hooks configured
- Jupyter, IPython, and debugging tools
- GPU support with CUDA 12.1

**Usage:**

```bash
docker-compose up -d pyrkm-dev
docker-compose exec pyrkm-dev bash
```

### 2. Jupyter Image (`pyrkm:jupyter`)

- Based on development image
- Pre-configured Jupyter Lab
- Accessible at <http://localhost:8888>

**Usage:**

```bash
docker-compose up -d pyrkm-jupyter
# Check logs for access token
docker-compose logs pyrkm-jupyter
```

### 3. Production Image (`pyrkm:prod`)

- Minimal image with only runtime dependencies
- Non-root user for security
- Optimized for deployment

**Usage:**

```bash
docker-compose up -d pyrkm-prod
```

### 4. CPU-Only Image (`pyrkm:cpu`)

- For environments without GPU
- PyTorch CPU version
- Smaller image size

**Usage:**

```bash
docker-compose -f docker-compose.cpu.yml up -d pyrkm-dev-cpu
```

### 5. Testing Image (`pyrkm:test`)

- Runs test suite automatically
- Generates coverage reports

**Usage:**

```bash
docker-compose run --rm pyrkm-test
```

## Persistent Volumes

Data is stored in named volumes:

- `pyrkm-data`: Dataset storage
- `pyrkm-models`: Trained model checkpoints
- `pyrkm-logs`: Training logs and TensorBoard data

To backup volumes:

```bash
docker run --rm -v pyrkm-data:/data -v $(pwd):/backup ubuntu tar czf /backup/pyrkm-data-backup.tar.gz /data
```

To restore volumes:

```bash
docker run --rm -v pyrkm-data:/data -v $(pwd):/backup ubuntu tar xzf /backup/pyrkm-data-backup.tar.gz -C /
```

## Common Workflows

### Training a Model

```bash
# Start dev container
make dev

# Inside container
python -c "
from pyrkm import ModelFactory, ConfigManager

config = ConfigManager()
config.update_model_config(model_name='my_model', n_visible=784, n_hidden=500)
config.update_training_config(learning_rate=0.01, max_epochs=10000)

model = ModelFactory.create_model('RKM')
# ... load data and train
"
```

### Running Tests

```bash
# Run all tests
make test

# Or manually
docker-compose run --rm pyrkm-dev pytest tests/ -v

# With coverage
docker-compose run --rm pyrkm-dev pytest tests/ --cov=src/pyrkm --cov-report=html
```

### Interactive Development with Jupyter

```bash
# Start Jupyter
make jupyter

# Access at http://localhost:8888
# Token is shown in logs: make logs
```

### Using Pre-commit Hooks

```bash
# Inside dev container
make shell

# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files

# Or use make command
make pre-commit
```

### Code Formatting and Linting

```bash
# Format code
make format

# Run linters
make lint
```

## Environment Variables

You can customize the environment using environment variables:

```bash
# In docker-compose.yml or docker-compose.override.yml
services:
  pyrkm-dev:
    environment:
      - NVIDIA_VISIBLE_DEVICES=0,1  # Use specific GPUs
      - PYTHONPATH=/app/src
      - CUDA_VISIBLE_DEVICES=0
      - LOG_LEVEL=DEBUG
```

## Troubleshooting

### GPU Not Detected

```bash
# Test GPU access
docker run --rm --runtime=nvidia pyrkm:dev python -c "import torch; print(torch.cuda.is_available())"

# Check NVIDIA runtime
docker run --rm --runtime=nvidia nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Permission Issues

```bash
# Fix volume permissions
docker-compose exec pyrkm-dev chown -R $(id -u):$(id -g) /app
```

### Out of Memory

```bash
# Limit GPU memory
docker-compose exec pyrkm-dev python -c "
import torch
torch.cuda.set_per_process_memory_fraction(0.5, 0)
"
```

### Container Won't Start

```bash
# Check logs
docker-compose logs pyrkm-dev

# Rebuild from scratch
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## Best Practices

1. **Use volumes for data**: Never store important data only inside containers
2. **Keep images small**: Use multi-stage builds and .dockerignore
3. **Use specific tags**: Don't rely on `latest` in production
4. **Security**: Run as non-root user in production
5. **Resource limits**: Set memory and CPU limits in production

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [NVIDIA Docker Documentation](https://github.com/NVIDIA/nvidia-docker)
- [PyTorch Docker Documentation](https://hub.docker.com/r/pytorch/pytorch)
