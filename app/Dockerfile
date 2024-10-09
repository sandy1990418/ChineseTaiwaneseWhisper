# Build stage
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY ../requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Remove unnecessary files
RUN find /root/.local -name '*.pyc' -delete && \
    find /root/.local -name '__pycache__' -delete

# Final stage
FROM nvidia/cuda:12.0.1-runtime-ubuntu20.04

WORKDIR /app 

# Install Python 3.10 and other system dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.10 \
    python3.10-distutils \
    python3.10-venv \
    libsndfile1 \
    ffmpeg \
    git \
    curl \
    && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.10 get-pip.py \
    && rm get-pip.py \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Copy Python packages from builder stage
COPY --from=builder /root/.local /root/.local

# Copy application files
COPY ../src/ ./src/
COPY ../scripts/ ./scripts/
COPY ../api_main.py .
COPY ../app/start.sh .

# Set PATH to include user-installed Python packages
ENV PATH=/usr/bin/:$PATH

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install FastAPI CLI
RUN pip install "fastapi[standard]"
RUN pip install uvicorn
RUN pip install --upgrade datasets evaluate

# Expose port
EXPOSE 8000


# Run the FastAPI server
# CMD ["python", "-m", "fastapi", "run", "api_main.py", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["uvicorn main:app --host 0.0.0.0 --port 80"]
# RUN chmod +x start.sh
# CMD ["python", "-m", "fastapi_cli", "run", "api_main.py", "--host", "0.0.0.0", "--port", "8000"]
# ["./start.sh"]

# Use the startup script
CMD ["python", "-m", "uvicorn", "api_main:app", "--host", "0.0.0.0", "--port", "8000"]
