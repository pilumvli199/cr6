# Dockerfile — Railway / Docker friendly, fixes TA-Lib build issues
FROM python:3.11-slim

# keep python output unbuffered
ENV PYTHONUNBUFFERED=1

# Install system build deps needed for compiling TA-Lib and Python extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    tar \
    gcc \
    make \
    pkg-config \
    libbz2-dev \
    liblzma-dev \
    libffi-dev \
    libssl-dev \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Build & install TA-Lib C library (only if you need native TA-Lib)
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz -O /tmp/ta-lib.tar.gz \
 && mkdir -p /tmp/ta-lib \
 && tar -xzf /tmp/ta-lib.tar.gz -C /tmp/ta-lib --strip-components=1 \
 && cd /tmp/ta-lib && ./configure --prefix=/usr && make && make install \
 && rm -rf /tmp/ta-lib /tmp/ta-lib.tar.gz

WORKDIR /app
COPY requirements.txt /app/requirements.txt

# Ensure pip/setuptools/wheel are recent
RUN python -m pip install --upgrade pip setuptools wheel

# Install numpy first so header files are available for building C extensions
RUN python -m pip install --no-cache-dir numpy==1.24.3

# Install other requirements except TA-Lib (we'll install TA-Lib separately)
RUN grep -i -v "^ta-lib" /app/requirements.txt > /tmp/reqs_no_talib.txt \
 && python -m pip install --no-cache-dir -r /tmp/reqs_no_talib.txt

# Install Python TA-Lib wrapper using the existing numpy and system TA-Lib C library
# Use --no-build-isolation so pip uses the already-installed numpy headers
RUN python -m pip install --no-cache-dir --no-build-isolation TA-Lib==0.4.28

# Copy application files
COPY . /app

# Expose port if web app (uncomment if needed)
# EXPOSE 8000

# Default command — change to your app entrypoint
CMD ["python", "railway_crypto_bot.py"]
