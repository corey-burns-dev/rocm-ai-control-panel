FROM ubuntu:jammy
ARG SD_BRANCH="master"

SHELL ["/bin/bash", "-c"]

ENV PORT=7860 \
	DEBIAN_FRONTEND=noninteractive \
	PYTHONUNBUFFERED=1 \
	PYTHONIOENCODING=UTF-8 \
	COMMANDLINE_ARGS=''

WORKDIR /opt

# Install base deps and clone AUTOMATIC1111 WebUI
RUN apt-get -y update && \
	apt-get install -y --no-install-recommends \
	libstdc++-12-dev ca-certificates wget gnupg2 gawk curl git \
	libglib2.0-0 apt-utils python3.10-venv python3-pip python-is-python3 \
	libgl1 && \
	git clone -b $SD_BRANCH https://github.com/AUTOMATIC1111/stable-diffusion-webui.git /sd && \
	true

WORKDIR /sd

# Clean up and install requirements
RUN apt-get autoremove -y && \
	apt-get clean -y && \
	rm -rf /var/lib/apt/lists/* && \
	python3 -m venv venv && \
	. venv/bin/activate && \
	pip install --upgrade pip wheel packaging && \
	pip install --pre --no-cache-dir \
	torch torchvision torchaudio \
	--index-url https://download.pytorch.org/whl/nightly/rocm6.3 && \
	pip install --no-cache-dir \
	gradio==3.41.2 \
	numpy==1.26.2 \
	Pillow==10.1.0 \
	requests==2.31.0 \
	tqdm==4.66.1 \
	psutil==5.9.6 \
	pytorch_lightning==1.9.0 \
	scikit-image==0.21.0 \
	fonts font-roboto timm==0.9.10 \
	accelerate==0.25.0 \
	transformers==4.36.2 \
	diffusers==0.25.1 \
	invisible-watermark==0.2.0 \
	pydantic==2.5.3 \
	kornia==0.7.0 \
	spandrel==0.3.4 \
	omegaconf==2.3.0 \
	safetensors==0.4.5 \
	einops==0.7.0 \
	jsonmerge==1.9.2 \
	clean-fid==0.1.35 \
	resize-right==0.0.2 \
	lark==1.1.9 \
	inflection==0.5.1 \
	GitPython==3.1.41 \
	torchsde==0.2.6 \
	httpcore==1.0.2 \
	fastapi==0.105.0 \
	httpx==0.26.0 \
	open-clip-torch==2.24.0 \
	piexif==1.1.3 || true

EXPOSE ${PORT}

VOLUME [ "/sd/configs","/sd/models", "/sd/outputs","/sd/extensions","/sd/plugins" ]

ENV TORCH_COMMAND="pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.3"

ENTRYPOINT ["/bin/bash", "-c", "source venv/bin/activate && python launch.py --listen --port 7860 --skip-torch-cuda-test"]
