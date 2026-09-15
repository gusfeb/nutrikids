FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements-render.txt .
RUN pip install --no-cache-dir -r requirements-render.txt

# Copy app code
COPY . .

# Create necessary directories
RUN mkdir -p static/uploads db dataset/raw_images dataset/images

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Run with gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "1"]
