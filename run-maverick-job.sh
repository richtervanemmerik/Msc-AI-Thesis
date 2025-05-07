# This script automates the setup and execution of the evaluation pipeline as a job in AzureML
# It performs the following steps:
# 1. Builds a Docker image required for the training process.
# 2. Downloads necessary datasets and resources from Azure Blob Storage.
#   - This is done because for now the job is not being authenticated properly to download the 
#       datasets from the blob storage.
# 3. Pushes the Docker image to an Azure Container Registry (ACR) for deployment.
# 4. Prepares the working directory by copying datasets, configuration files, and source code.
# 5. Extracts specific files from compressed datasets as needed.
# 6. Launches the training job using a Python script or command-line tool.
# 7. Cleans up temporary files and directories after the training job is initiated.
#
# This script must be executed from the same directory where it resides.

# echo "Logging in to Azure..."
# az acr login --name kpmgnldankfwepacr && \

# # Build the Docker image
# echo "Building Docker image..."
# docker build -f Dockerfile -t maverick-training:latest . && \
# # Push the Docker image to Azure Container Registry (ACR)
# docker tag maverick-training:latest kpmgnldankfwepacr.azurecr.io/richter-master-thesis/maverick-training:latest && \
# docker push kpmgnldankfwepacr.azurecr.io/richter-master-thesis/maverick-training:latest && \

python azure_submit.py