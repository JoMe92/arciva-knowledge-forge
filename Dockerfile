# Use PyTorch base image with CUDA support
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy DVC configuration
COPY .dvc /app/.dvc
COPY .dvcignore /app/.dvcignore

# Copy project files
# We accept that this might copy data if it's not in .dockerignore, but typically data is ignored or we are careful.
# Ideally we copy specific files, but COPY . . is standard for simple setups.
COPY . .

# Set env var for DVC
ENV DVC_NO_ANALYTICS=1

# Default command
CMD ["python", "finetune.py"]
