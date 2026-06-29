# VARUNA — AI Digital Twin of India's Climate · Hugging Face Docker Space
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    HF_HOME=/tmp/hf \
    MPLCONFIGDIR=/tmp/mpl \
    XDG_CACHE_HOME=/tmp/cache

WORKDIR /app

# System libs for netCDF / HDF5 / scientific stack
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev libnetcdf-dev build-essential && \
    rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch first (keeps the image small), then the rest
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

COPY . .

# the rebuildable anomaly cache is written at runtime — keep paths writable
RUN mkdir -p data/processed outputs && chmod -R 777 data outputs

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
