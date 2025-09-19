# Declare all phony targets
.PHONY: install format lint clean test all

# Default target
.DEFAULT_GOAL := all

# Variables
SRC_PROJECT_NAME ?= src
SRC_TESTS ?= tests
FAIL_UNDER ?= 8

# Install project dependencies
install:
	@echo "Installing dependencies..."
	@uv sync
	@echo "Dependencies installed! ✅"

# Code formatting
format:
	@echo "Formatting code..."
	@isort .
	@black .
	@echo "Formatting completed! ✅"

# Check code
lint:
	@echo "Running isort..."
	@isort .
	@isort $(SRC_PROJECT_NAME) $(SRC_TESTS) --check
	@echo "Running Black..."
	@black .
	@black --check .
	@echo "Running Mypy..."
	@mypy $(SRC_PROJECT_NAME) $(SRC_TESTS)
	@echo "Running Flake8..."
	@flake8 $(SRC_PROJECT_NAME) $(SRC_TESTS)
	@echo "Running Ruff..."
	@ruff check $(SRC_PROJECT_NAME) $(SRC_TESTS)
	@echo "Running Complexipy..."
	@complexipy .
	@echo "Running Pylint..."
	@pylint --fail-under=$(FAIL_UNDER) $(SRC_PROJECT_NAME) $(SRC_TESTS)
	@echo "Lint completed! ✅"

# Clean cache and temporary files
clean:
	@echo "Cleaning cache and temporary files..."
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@find . -type d -name .pytest_cache -exec rm -rf {} +
	@find . -type d -name .mypy_cache -exec rm -rf {} +
	@find . -type d -name .ruff_cache -exec rm -rf {} +
	@find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
	@rm -f .coverage
	@echo "Cleaning completed! ✅"

# Test the code (it is assumed tests are in the "tests" folder and start with "test_")
test:
	@echo "Checking if tests directory exists..."
	@if [ -d "$(SRC_TESTS)" ] && [ $$(find $(SRC_TESTS) -name "test_*.py" | wc -l) -gt 0 ]; then \
		echo "Running tests..."; \
		pytest tests --cov src; \
		echo "Tests passed! ✅"; \
	else \
		echo "No tests directory found or no test files. Skipping tests."; \
	fi

# Run all workflows
all: install format lint test clean
	@echo "All tasks completed! ✅"