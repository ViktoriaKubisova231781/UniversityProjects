from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    KubernetesOnlineDeployment,
    CodeConfiguration,
    ResourceRequirementsSettings,
    ResourceSettings,
    OnlineRequestSettings 
)
from azure.identity import InteractiveBrowserCredential

# ─── Config ─────────────────────────────────────────────
SUBSCRIPTION_ID = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
RESOURCE_GROUP_NAME = "buas-y2"
WORKSPACE_NAME = "CV6-2025"
ENDPOINT_NAME = "viki-unet-endpoint"
DEPLOYMENT_NAME = "viki-model-deployment"
MODEL = "unet_model_viki:1"
ENVIRONMENT = "iris:0.8.0"
COMPUTE = "adsai-lambda-0"

# ─── Main ───────────────────────────────────────────────
def main():
    ml_client = MLClient(
        InteractiveBrowserCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP_NAME,
        workspace_name=WORKSPACE_NAME
    )

    print(f"Starting deployment: {DEPLOYMENT_NAME} to {ENDPOINT_NAME}")

    deployment = KubernetesOnlineDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        model=MODEL,
        environment=ENVIRONMENT,
        code_configuration=CodeConfiguration(
            code="ml_deployment",
            scoring_script="score_viki.py"
        ),
        instance_count=1,

        request_settings=OnlineRequestSettings(
            request_timeout_ms=120000,  # 2 minutes max per request
            max_concurrent_requests_per_instance=1,
            max_queue_wait_ms=60000     # 1 min max queue wait
        ),

        resources=ResourceRequirementsSettings(
            requests=ResourceSettings(cpu="8", memory="16Gi"),
            limits=ResourceSettings(cpu="8", memory="16Gi")
        )
    )

    # Deploy the model
    ml_client.online_deployments.begin_create_or_update(deployment).result()
    print(f"Deployment '{DEPLOYMENT_NAME}' created.")

    # Route all traffic to this deployment
    endpoint = ml_client.online_endpoints.get(name=ENDPOINT_NAME)
    endpoint.traffic = {DEPLOYMENT_NAME: 100}
    ml_client.begin_create_or_update(endpoint).result()
    print(f"100% traffic routed to '{DEPLOYMENT_NAME}'.")

    # Display the scoring URI and logs
    print(f"Scoring URI: {endpoint.scoring_uri}")
    logs = ml_client.online_deployments.get_logs(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        lines=30
    )
    print(f"Logs:\n{logs}")

if __name__ == "__main__":
    main()
