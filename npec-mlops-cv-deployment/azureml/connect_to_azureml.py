# connect_to_azureml.py

from azure.identity import AzureCliCredential
from azure.ai.ml import MLClient

# Replace these values with your actual Azure settings
subscription_id = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
resource_group = "buas-y2"
workspace_name = "CV6-2025"

# Connect to Azure ML
ml_client = MLClient(
    AzureCliCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name
)

# Test: list models in the workspace
models = ml_client.models.list()
for model in models:
    print(f"Model: {model.name}, Version: {model.version}")
