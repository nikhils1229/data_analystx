# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for data analysis
RUN pip install --no-cache-dir \
    pandas \
    matplotlib \
    seaborn \
    requests \
    beautifulsoup4 \
    duckdb \
    lxml

# The container will run a command passed to it by the Docker SDK
CMD ["python"]
