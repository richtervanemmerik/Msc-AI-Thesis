# Base image for Azure Machine Learning
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Install base requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY base_requirements.txt /tmp/base_requirements.txt
RUN pip install --no-cache-dir -r /tmp/base_requirements.txt

RUN python -m spacy download en_core_web_sm
RUN python -m spacy download en_core_web_lg
RUN python -m spacy download en_core_web_trf

RUN mkdir -p /maverick/data
COPY ./data /maverick/data

# Default command
CMD ["/bin/bash"]