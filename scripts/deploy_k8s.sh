#!/usr/bin/env bash
set -euo pipefail

# deploy_k8s.sh
# Simple helper to build images, optionally push to a registry, and deploy k8s manifests.

REGISTRY=""
PUSH=false
NAMESPACE="rocm-ai"
ENABLE_SD=false

usage() {
  cat <<EOF
Usage: $0 [--registry registry.example.com] [--push] [--namespace <ns>] [--enable-sd]

Options:
  --registry    Registry host (e.g. registry.example.com/user)
  --push        Tag and push built images to the registry
  --namespace   Kubernetes namespace to deploy (default: rocm-ai)
  --enable-sd   Also deploy the SD WebUI (sd-webui.yaml). By default it's commented out.

This script will:
  - Build the FastAPI image (rocm-fastapi:latest)
  - If --registry is provided, tag and push images
  - Apply kustomize manifests under k8s/
  - If registry provided and pushed, update the Deployment images
  - Show rollout status and pod list

Note: For multi-node k3s clusters, push images to a registry or load into containerd on nodes.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry)
      shift; REGISTRY="$1"; ;;
    --push)
      PUSH=true; ;;
    --namespace)
      shift; NAMESPACE="$1"; ;;
    --enable-sd)
      ENABLE_SD=true; ;;
    -h|--help)
      usage; exit 0; ;;
    *)
      echo "Unknown arg: $1"; usage; exit 1; ;;
  esac
  shift
done

echo "Namespace: $NAMESPACE"
if [[ -n "$REGISTRY" ]]; then
  echo "Registry: $REGISTRY"
fi

# 1) Build FastAPI image
echo "Building FastAPI image..."
docker build -t rocm-fastapi:latest ./fastapi

# Optionally tag and push
if [[ -n "$REGISTRY" ]]; then
  FASTAPI_IMG="$REGISTRY/rocm-fastapi:latest"
  echo "Tagging $FASTAPI_IMG"
  docker tag rocm-fastapi:latest "$FASTAPI_IMG"
  if [[ "$PUSH" = true ]]; then
    echo "Pushing $FASTAPI_IMG"
    docker push "$FASTAPI_IMG"
  fi
fi

# Optionally build SD WebUI if enabling
if [[ "$ENABLE_SD" = true ]]; then
  echo "Building SD WebUI image (local tag stable-diffusion-webui-rocm:6.4)..."
  docker build -t stable-diffusion-webui-rocm:6.4 ./stable-diffusion-webui-rocm || true
  if [[ -n "$REGISTRY" ]]; then
    SDIMG="$REGISTRY/stable-diffusion-webui-rocm:6.4"
    docker tag stable-diffusion-webui-rocm:6.4 "$SDIMG" || true
    if [[ "$PUSH" = true ]]; then
      docker push "$SDIMG" || true
    fi
  fi
fi

# 2) Apply kustomize manifests
echo "Applying kustomize manifests..."
# If sd enabled, ensure kustomization includes sd-webui
if [[ "$ENABLE_SD" = true ]]; then
  echo "Enabling sd-webui in kustomization.yaml (if commented)"
  # Attempt to uncomment sd-webui entry if commented out
  sed -i 's/^\s*# - sd-webui.yaml\s*/  - sd-webui.yaml/' k8s/kustomization.yaml || true
fi

kubectl apply -k k8s/

# 3) If registry provided and pushed, patch deployments to use registry images
if [[ -n "$REGISTRY" ]]; then
  echo "Updating deployments to use registry images..."
  kubectl -n "$NAMESPACE" set image deployment/fastapi fastapi=${FASTAPI_IMG} --record || true
  if [[ "$ENABLE_SD" = true ]]; then
    kubectl -n "$NAMESPACE" set image deployment/sd-webui sd-webui=${SDIMG} --record || true
  fi
fi

# 4) Wait for rollout
echo "Waiting for deployments to become ready..."
kubectl -n "$NAMESPACE" rollout status deployment/fastapi --timeout=120s || true
if [[ "$ENABLE_SD" = true ]]; then
  kubectl -n "$NAMESPACE" rollout status deployment/sd-webui --timeout=300s || true
fi

# 5) Show pods and services
kubectl -n "$NAMESPACE" get pods -o wide
kubectl -n "$NAMESPACE" get svc

echo "Done. If your cluster uses containerd (k3s), ensure images are available to nodes or use a registry."

echo "Helpful next steps:"
echo "  kubectl logs -n $NAMESPACE -l app=fastapi -f"
echo "  kubectl port-forward -n $NAMESPACE svc/fastapi 5000:5000"

exit 0
