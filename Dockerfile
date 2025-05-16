# Use the official PyTorch image with CUDA (or switch to CPU-only if needed)
FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .

# Install system dependencies and Python packages
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY . .

# Expose Flask port
EXPOSE 5000

# Run the Flask app
CMD ["python", "\clip_score.py"]
