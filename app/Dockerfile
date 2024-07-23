# Dockerfile (https://hub.docker.com/r/pytorch/pytorch/tags?page=2&page_size=&name=&ordering=)

FROM  pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel

WORKDIR /app

# RUN DEBIAN_FRONTEND=noninteractive apt install -y tzdata
# Base system tools
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -yqq install tzdata
# Set build arguments
# ARG DEBIAN_FRONTEND=noninteractive
# ARG TZ=Etc/UTC

# Set environment variables
# ENV TZ=${TZ}


# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy project files
COPY . /app
COPY src ./src
COPY app/.env .
COPY scripts ./scripts

# Expose port
EXPOSE 8000

# Run FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]