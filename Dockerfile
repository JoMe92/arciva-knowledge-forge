# Use PyTorch base image with CUDA support
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies (git is often needed for pip installing from git)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set env var for DVC to work in container if needed (e.g. no analytics)
ENV DVC_NO_ANALYTICS=1

# Default command
CMD ["python", "finetune.py"]
