# Pull official base image
FROM python:3.11.4-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /usr/src/app

# Install dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements file and install Python dependencies
COPY requirements.txt /usr/src/app/
RUN pip install --no-cache-dir  -r requirements.txt --timeout 1000

# Copy project files
COPY . /usr/src/app/

# Copy and fix line endings in entrypoint script, then make it executable
COPY ./entrypoint.sh /usr/src/app/
RUN sed -i 's/\r$//g' /usr/src/app/entrypoint.sh
RUN chmod +x /usr/src/app/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/usr/src/app/entrypoint.sh"]
