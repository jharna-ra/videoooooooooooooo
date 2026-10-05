# GPU backend. Works on Hugging Face Spaces (Docker + GPU), RunPod, Modal, any NVIDIA host.
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN git clone -q https://github.com/sczhou/ProPainter /app/ProPainter \
    && pip install --no-cache-dir -r /app/ProPainter/requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pipeline.py server.py ./
ENV PROPAINTER_DIR=/app/ProPainter DATA_DIR=/tmp/remover
EXPOSE 7860
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-7860}"]
