"""Upload a Keras model to Azure ML Workspace."""

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import InteractiveBrowserCredential

SUBSCRIPTION_ID = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
RESOURCE_GROUP = "buas-y2"
WORKSPACE_NAME = "CV6-2025"
SRC_PATH = r"C:\Users\vikku\OneDrive\Documenten\Buas\2024-25d-fai2-adsai-group-CV6\CV_6\models\12_viktoria_231781_unet_model_256px.h5"
NAME = "unet_model_viki"


def main() -> None:
    """Authenticate and upload model to Azure ML workspace."""
    ml_client = MLClient(
        credential=InteractiveBrowserCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )

    model = Model(
        path=SRC_PATH,
        name=NAME,
        description="Viki UNet model for plant segmentation",
        type="custom_model",
    )

    ml_client.models.create_or_update(model)
    print(f"Model '{NAME}' uploaded from {SRC_PATH}")


if __name__ == "__main__":
    main()
