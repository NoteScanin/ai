FROM python:3.10-slim

# Install system dependencies untuk OpenCV & image processing
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces mengharuskan user non-root
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install Python dependencies dulu (cache layer)
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy source code
COPY --chown=user . /app

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=microsoft/trocr-base-handwritten
ENV MAX_FILE_SIZE=10485760
ENV BATCH_SIZE=4

# Hugging Face Spaces WAJIB port 7860
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
