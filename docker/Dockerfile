FROM registry.gitlab.com/brienza1/empower_docker:latest

WORKDIR /app

# Install face recognition dependencies (including system cmake)
RUN apt-get update && \
    apt-get install -y build-essential cmake python3-dev python3-pip libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev && \
    pip install --upgrade pip

# Add Python dependencies (remove cmake from requirements.txt)
COPY docker/requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/