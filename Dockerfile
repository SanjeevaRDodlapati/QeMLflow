# Multi-stage Dockerfile for ChemML
# Stage 1: Base environment with system dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r chemml && useradd -r -g chemml chemml
WORKDIR /app
RUN chown chemml:chemml /app

# Stage 2: Development environment
FROM base as development

# Install development dependencies
COPY pyproject.toml .
RUN pip install -e ".[dev]"

# Copy source code
COPY --chown=chemml:chemml . .

# Switch to non-root user
USER chemml

# Default command for development
CMD ["python", "-c", "import src; print('ChemML development environment ready')"]

# Stage 3: Production environment
FROM base as production

# Install only production dependencies
COPY pyproject.toml .
RUN pip install .

# Copy only necessary files
COPY --chown=chemml:chemml src/ ./src/
COPY --chown=chemml:chemml README.md .
COPY --chown=chemml:chemml LICENSE .

# Switch to non-root user
USER chemml

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src; print('ChemML is healthy')" || exit 1

# Default command
CMD ["python", "-c", "import src; print('ChemML production environment ready')"]

# Stage 4: Jupyter notebook environment
FROM base as notebook

# Install Jupyter and notebook dependencies
COPY pyproject.toml .
RUN pip install -e ".[dev,jupyter]" jupyter

# Copy notebooks and source code
COPY --chown=chemml:chemml . .

# Switch to non-root user
USER chemml

# Expose Jupyter port
EXPOSE 8888

# Default command for notebook server
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
