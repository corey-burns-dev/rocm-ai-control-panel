# ROCm AI Lab

ROCm AI Lab is a local AMD-focused AI control panel for managing Ollama, Stable Diffusion WebUI, and a small FastAPI backend on a ROCm-capable machine.

## Highlights

- AMD GPU-oriented local AI stack
- FastAPI backend plus browser control panel
- Stable Diffusion and Ollama services in one compose flow

## Requirements

- AMD GPU with ROCm support
- Working ROCm drivers on the host

## Quick start

```bash
docker compose up -d --build
```

The root `index.html` provides the lightweight control panel once the services are up.
