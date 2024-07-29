# Dockerfile (https://hub.docker.com/r/pytorch/pytorch/tags?page=2&page_size=&name=&ordering=)

FROM  pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel

WORKDIR /ChineseTaiwaneseWhisper

# RUN DEBIAN_FRONTEND=noninteractive apt install -y tzdata
# Base system tools
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -yqq install tzdata

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install  --no-cache-dir  -r requirements.txt

# Copy project files
COPY . /ChineseTaiwaneseWhisper
COPY src ./src
COPY app/.env ./app
COPY scripts ./scripts
COPY app/main.py .

# Expose port
EXPOSE 8000

# Run FastAPI application
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["echo", "$HOME"]
