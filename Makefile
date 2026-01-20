.PHONY: help build build-cpu dev jupyter test clean

# Display help information
help:
	@echo "PyRKM Docker Commands:"
	@echo "  make build          - Build Docker images with GPU support"
	@echo "  make build-cpu      - Build Docker images for CPU-only"
	@echo "  make dev            - Start development container with GPU"
	@echo "  make dev-cpu        - Start development container CPU-only"
	@echo "  make jupyter        - Start Jupyter Lab with GPU"
	@echo "  make jupyter-cpu    - Start Jupyter Lab CPU-only"
	@echo "  make test           - Run tests in container"
	@echo "  make tensorboard    - Start TensorBoard"
	@echo "  make clean          - Remove all containers and images"
	@echo "  make clean-volumes  - Remove all volumes (WARNING: deletes data)"
	@echo "  make shell          - Open shell in dev container"
	@echo "  make logs           - View logs from all containers"

# Build GPU-enabled images
build:
	docker-compose build

# Build CPU-only images
build-cpu:
	docker-compose -f docker-compose.cpu.yml build

# Start development container with GPU
dev:
	docker-compose up -d pyrkm-dev
	docker-compose exec pyrkm-dev bash

# Start development container CPU-only
dev-cpu:
	docker-compose -f docker-compose.cpu.yml up -d pyrkm-dev-cpu
	docker-compose -f docker-compose.cpu.yml exec pyrkm-dev-cpu bash

# Start Jupyter Lab with GPU
jupyter:
	docker-compose up -d pyrkm-jupyter
	@echo "Jupyter Lab is running at http://localhost:8888"
	@echo "Check logs for the token: make logs"

# Start Jupyter Lab CPU-only
jupyter-cpu:
	docker-compose -f docker-compose.cpu.yml up -d pyrkm-jupyter-cpu
	@echo "Jupyter Lab is running at http://localhost:8888"

# Run tests
test:
	docker-compose run --rm pyrkm-test

# Start TensorBoard
tensorboard:
	docker-compose up -d tensorboard
	@echo "TensorBoard is running at http://localhost:6006"

# Open shell in dev container
shell:
	docker-compose exec pyrkm-dev bash

# View logs
logs:
	docker-compose logs -f

# Stop all containers
stop:
	docker-compose down
	docker-compose -f docker-compose.cpu.yml down

# Clean up containers and images
clean:
	docker-compose down --rmi all
	docker-compose -f docker-compose.cpu.yml down --rmi all

# Clean up volumes (WARNING: this will delete your data)
clean-volumes:
	docker-compose down -v
	docker-compose -f docker-compose.cpu.yml down -v

# Install pre-commit hooks in dev container
pre-commit:
	docker-compose exec pyrkm-dev pre-commit install
	docker-compose exec pyrkm-dev pre-commit run --all-files

# Format code in dev container
format:
	docker-compose exec pyrkm-dev black src/ tests/
	docker-compose exec pyrkm-dev isort src/ tests/

# Run linting
lint:
	docker-compose exec pyrkm-dev flake8 src/ tests/
	docker-compose exec pyrkm-dev mypy src/

# Build production image
build-prod:
	docker-compose build pyrkm-prod

# Run production container
prod:
	docker-compose up -d pyrkm-prod
