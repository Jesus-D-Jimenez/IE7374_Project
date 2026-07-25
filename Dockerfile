# Reproducible CPU environment for the whole pipeline.
#
#   docker build -t black-box-lm .
#   docker run --rm -v "$(pwd)/outputs:/app/outputs" black-box-lm
#
# The default command is the milestone entry point (python src/model_runner.py). Mounting
# ./outputs writes the generated samples back to the host. Mounting a Hugging Face cache
# (-v hf-cache:/app/.cache/huggingface) avoids re-downloading the model on every run.
FROM python:3.11-slim

# Fail fast, no .pyc clutter, unbuffered logs so `docker run` streams progress live.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Dependencies first so the (slow) install layer is cached across code edits.
# The CPU-only wheel index keeps the image ~2 GB smaller than the default CUDA build.
COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.1" \
 && pip install -r requirements.txt

COPY . .
RUN pip install --no-deps -e .

# Model weights download on first use; bake them in with
#   RUN python -c "from transformer_lens import HookedTransformer as H; H.from_pretrained('pythia-160m', device='cpu')"
# if you prefer a fully offline image (adds ~380 MB).

CMD ["python", "src/model_runner.py"]
