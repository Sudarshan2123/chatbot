FROM python:3.11-bookworm

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Protocol buffers and gRPC configuration
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV GRPC_VERBOSITY=error

# Install essential system dependencies
# We keep build-essential for any python packages that need to compile C extensions
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get upgrade -y openssl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose the application port
EXPOSE 5050

# Set environment variables for the application
ENV HOST=0.0.0.0
ENV PORT=5050

# Start the application
CMD ["uvicorn", "app:main_app", "--host", "0.0.0.0", "--port", "5050"]