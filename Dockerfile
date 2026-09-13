# Multi-stage build for the hackathon submission
FROM python:3.11-slim AS base

# System deps for headless MuJoCo + OpenVINO + OpenGL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libglfw3 \
    libosmesa6-dev \
    libglew-dev \
    ffmpeg \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

ENV MUJOCO_GL=osmesa
ENV DISPLAY=:0

WORKDIR /app

# Clone SO-101 assets
RUN git clone --depth 1 https://github.com/TheRobotStudio/SO-ARM100.git third_party/SO-ARM100

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Build scene (Mock for the prompt)
# RUN python scene/build_scene.py --headless-build
RUN echo "Scene build skipped for standalone container test"

EXPOSE 8000

CMD ["uvicorn", "dashboard.server:app", "--host", "0.0.0.0", "--port", "8000"]
