# Dockerfile for pyiwfm with web visualization
#
# For full Linux support including HEC-DSS, use:
#   docker build -f dss-build/Dockerfile -t pyiwfm-dss .
#
# Build: docker build -t pyiwfm .
# Run:   docker run -p 8080:8080 -v /path/to/model:/model pyiwfm

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy pyiwfm package
COPY . /app/

# Version arg — .git is excluded by .dockerignore, so hatch-vcs cannot
# detect the version automatically.  Pass the version at build time:
#   docker build --build-arg VERSION=1.0.4 -t pyiwfm .
# Falls back to "0.0.0" for local builds without the arg.
ARG VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION}

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
