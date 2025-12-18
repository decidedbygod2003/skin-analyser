# # Use Python 3.11 slim image for smaller size
# FROM python:3.11-slim

# # Set working directory
# WORKDIR /app

# # Install system dependencies required for OpenCV
# # RUN apt-get update && apt-get install -y \
# #     libgl1-mesa-glx \
# #     libglib2.0-0 \
# #     libsm6 \
# #     libxext6 \
# #     libxrender-dev \
# #     && rm -rf /var/lib/apt/lists/*

# # Copy requirements first for better caching
# COPY requirements.txt .

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy application files
# COPY . .

# # Create directory for temporary files
# RUN mkdir -p /tmp/streamlit

# # Expose Streamlit default port
# EXPOSE 8501

# # Set environment variables
# ENV STREAMLIT_SERVER_PORT=8501
# ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
# ENV STREAMLIT_SERVER_HEADLESS=true
# ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# # Health check
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#     CMD curl -f http://localhost:8501/_stcore/health || exit 1

# # Run the application
# CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install minimal system dependencies (NO libGL needed for headless OpenCV)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better Docker cache)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose FastAPI port
EXPOSE 8501

# Environment variables (optional but good practice)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check for API
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/docs || exit 1

# Run FastAPI app on port 8501
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8501"]

