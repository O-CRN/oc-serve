FROM rayproject/ray:2.48.0-py312-cu128

WORKDIR /

USER root

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx numactl git ffmpeg \
    libsndfile1 espeak espeak-ng \
    && rm -rf /var/lib/apt/lists/*

USER ray

WORKDIR /app

RUN pip install --upgrade pip --no-cache-dir

COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir

RUN uv pip install "sglang[all]>=0.5.3rc0" --index-strategy unsafe-best-match --system

RUN MAX_JOBS=4 pip install flash-attn==2.8.2 --no-build-isolation --no-cache-dir

COPY . .
