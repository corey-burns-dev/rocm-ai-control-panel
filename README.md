# ROCm AI Lab

This repository contains a local ROCm-based AI control panel and services for running Stable Diffusion (AUTOMATIC1111 WebUI), an Ollama LLM server, and a FastAPI backend that ties the UI together. The project is designed for AMD GPU systems (ROCm).

## Contents

- `docker-compose.yml` — Compose setup for local development (Ollama, FastAPI, sd-webui)
- `fastapi/` — FastAPI backend code and Dockerfile
- `stable-diffusion-webui-rocm/` — WebUI build context, helper scripts and Dockerfile
- `index.html` — A lightweight browser control panel for image generation, chat, and SVG tools

## Requirements

- AMD GPU with ROCm support (tested on AMD Radeon RX 9070)
- ROCm drivers installed on the host (tested on ROCm 6.4.4-1.1 on Arch Linux)

## Quick Start

1. Ensure ROCm drivers are installed on the host and the AMD GPU is accessible
2. Start services:

   ```fish
   docker compose up -d --build
   ```

3. Open the control panel in a browser — the UI is served from `index.html` in the repo root

## Notes

- The compose file mounts host device nodes (`/dev/kfd`, `/dev/dri`) and sets ROCm environment variables
- `sd-webui` uses flags like `--medvram` and `--opt-split-attention` for lower VRAM usage

## Features

- FastAPI backend with endpoints to manage models, generate images, and optional SVG/logo generators
- Stable Diffusion WebUI for image generation
- Ollama LLM server for chat capabilities

## Security & Production Notes

- GPU containers run with privileged access to device nodes; consider hardening for production
- Use an authenticated ingress or VPN when exposing services externally
