#FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# if it will not work, install via apt!


ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Install essentials + OpenBLAS/LAPACK properly
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates python3 python3-venv python3-distutils build-essential \
    libgl1 libglib2.0-0 libopenblas-dev liblapack-dev libjpeg-dev zlib1g-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

# Upgrade pip and setuptools before numpy
RUN pip install --upgrade pip setuptools wheel

COPY environments.sh .
COPY requirements.txt .

RUN sh environments.sh

# Copy your app
COPY . .

RUN sh download_checkpoints.sh

RUN apt-get update && apt-get install -y git git-lfs && git lfs install && \
    git clone https://huggingface.co/facebook/wav2vec2-base-960h /workspace/checkpoints/wav2vec2-base-960h && \
    git clone https://huggingface.co/r-f/wav2vec-english-speech-emotion-recognition /workspace/checkpoints/wav2vec-english-speech-emotion-recognition


# Install runpod deps last
RUN pip install runpod boto3 requests

RUN pip install numpy==1.24.4

# Optional: quick check to verify numpy availability
RUN python3 -c "import torch, numpy; print('NumPy OK:', numpy.__version__, '| Torch:', torch.__version__)"

RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
    mkdir -p /root/.cache/huggingface/transformers && \
    wget -q https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth -O /root/.cache/torch/hub/checkpoints/s3fd-619a316812.pth && \
    wget -q https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip -O /root/.cache/torch/hub/checkpoints/2DFAN4-cd938726ad.zip && \
    python3 -c "from transformers import Wav2Vec2Model; Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')"

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PYTHONUNBUFFERED=1
CMD ["python3", "-u", "runpod_handler.py"]
