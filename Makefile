# Makefile for notebook/markdown exporter Docker image

# Variables
IMAGE_NAME := notebook-exporter
CONTAINER_NAME := notebook-exporter-run
DOCKERFILE := tools/Dockerfile.exporter
WORKDIR := $(shell pwd)
PYTHON_CMD := python tools/export_all.py

# Default target
.PHONY: all
all: build run

# Build the Docker image
.PHONY: build
build:
	docker build -f $(DOCKERFILE) -t $(IMAGE_NAME) .

# Run the exporter (mount current repo for live access)
.PHONY: run
run:
	docker run --rm -v $(WORKDIR):/site $(IMAGE_NAME)

# Run interactively for debugging
.PHONY: shell
shell:
	docker run --rm -it -v $(WORKDIR):/site --entrypoint /bin/bash $(IMAGE_NAME)

# Rebuild from scratch (no cache)
.PHONY: rebuild
rebuild:
	docker build --no-cache -f $(DOCKERFILE) -t $(IMAGE_NAME) .

# Clean up any stopped containers or dangling images
.PHONY: clean
clean:
	docker container prune -f
	docker image prune -f

# Force re-run without rebuild
.PHONY: rerun
rerun: run
