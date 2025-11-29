# Base Image: Official Python 3.10 Slim (Lightweight)
FROM python:3.10-slim

# Metadata
LABEL maintainer="NeuroLoft Lab <ecobrainstorming@gmail.com>"
LABEL description="NeuroLoft API Container"

# Set working directory inside the container
WORKDIR /code

# Prevent Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE=1
# Prevent Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching layers)
COPY requirements.txt .

# Install dependencies
# Using CPU-only PyTorch to keep image size small (~700MB vs 3GB+)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the application
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
