FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies (adjust path if your requirements are elsewhere)
COPY Homogeneous_Patches_CNN_v2/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt

# Copy project files
COPY . /app

# Default command: drop to a shell. Replace with your app entrypoint as needed.
CMD ["bash"]
