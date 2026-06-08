from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import InteractiveBrowserCredential

# Workspace config
SUBSCRIPTION_ID = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
RESOURCE_GROUP = "buas-y2"
WORKSPACE_NAME = "CV6-2025"

# Local paths
BASE_LOCAL_PATH = r"C:\Users\vikku\OneDrive\Documenten\Buas\2024-25d-fai2-adsai-ViktoriaKubisova231781\CV_6\data\original_dataset\patches"

# Create ML client
ml_client = MLClient(
    credential=InteractiveBrowserCredential(),
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME
)

# Data assets to register
splits = ["train", "val", "test"]

for split in splits:
    local_path = f"{BASE_LOCAL_PATH}\\{split}"
    
    data_asset = Data(
        name=f"plant_data_{split}_patches",
        version="1.0.0",
        description=f"{split} dataset for patch-based root segmentation",
        path=local_path,
        type=AssetTypes.URI_FOLDER,
    )
    
    ml_client.data.create_or_update(data_asset)
    print(f"Registered: {split}")
