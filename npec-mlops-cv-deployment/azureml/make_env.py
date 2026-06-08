"""
Create or update Azure Machine Learning Environment (SDK v2).
"""
 
from azure.identity import InteractiveBrowserCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
 
SUBSCRIPTION_ID = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
RESOURCE_GROUP = "buas-y2"
WORKSPACE_NAME = "CV6-2025"
ENV_NAME = "iris"
ENV_YAML_PATH = "./ml_deployment/environment.yml"
VERSION = "0.9.0"
 
def main():
   
    ml_client = MLClient(
        credential=InteractiveBrowserCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME
    )
 
    # Create environment from YAML
    env = Environment(
        name=ENV_NAME,
        conda_file=ENV_YAML_PATH,
        description="Environment for Iris classification model training and inference",
        image="mcr.microsoft.com/azureml/minimal-ubuntu22.04-py39-cpu-inference:latest",
        version=VERSION
    )
 
    # Register environment
    register_env = ml_client.environments.create_or_update(env)
    print(f"Environment '{ENV_NAME}' registered successfully in workspace '{WORKSPACE_NAME}'.")
 
if __name__ == "__main__":
    main()
 