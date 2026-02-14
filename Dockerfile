# Dockerfile for pyiwfm with web visualization
#
# Build: docker build -t pyiwfm .
# Run:   docker run -p 8080:8080 -v /path/to/model:/model pyiwfm

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy pyiwfm package
COPY . /app/

# Install pyiwfm with all dependencies including web visualization
RUN pip install --no-cache-dir -e ".[all]" || \
    pip install --no-cache-dir -e ".[gis,viz,webapi]"

# Create directory for model data
RUN mkdir -p /model

# No server-side rendering needed (client-side vtk.js)

# Expose the web viewer port
EXPOSE 8080

# Copy the startup script
COPY docker-entrypoint.py /app/docker-entrypoint.py

# Default command - start the web viewer
CMD ["python", "/app/docker-entrypoint.py"]
