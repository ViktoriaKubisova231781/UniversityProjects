"""Create or update Azure Machine Learning Endpoint.

This script creates or updates a Kubernetes-based online endpoint
in Azure Machine Learning. It uses the Azure ML SDK to define
the endpoint and deploy it to a specified compute target.

Example:
    $ python endpoint_creation.py
"""

from azure.ai.ml import MLClient
from azure.identity import InteractiveBrowserCredential
from azure.ai.ml.entities import KubernetesOnlineEndpoint

SUBSCRIPTION_ID = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
RESOURCE_GROUP_NAME = "buas-y2"
WORKSPACE_NAME = "CV6-2025"
ENDPOINT_NAME = "viki-unet-endpoint"
COMPUTE = "adsai-lambda-0"


def main():
    """Register an endpoint using 'KubernetesOnlineEndpoint'."""
    ml_client = MLClient(
        InteractiveBrowserCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP_NAME,
        workspace_name=WORKSPACE_NAME
    )

    endpoint = KubernetesOnlineEndpoint(
        name=ENDPOINT_NAME,
        compute=COMPUTE,
        description="Endpoint for the model",
        auth_mode="key"
    )

    ml_client.online_endpoints.begin_create_or_update(endpoint).result()


if __name__ == "__main__":
    main()
