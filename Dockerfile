# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt ./
# Install Python dependencies for the main app AND for data analysis
RUN pip install --no-cache-dir -r requirements.txt \
    pandas \
    matplotlib \
    seaborn \
    requests \
    beautifulsoup4 \
    duckdb \
    lxml

# Copy the rest of the application code
COPY . .

# Tell Docker what port the app will run on
EXPOSE 10000

# This command starts the web server
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}
