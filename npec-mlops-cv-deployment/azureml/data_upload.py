"""Upload local data to AzureML workspaceblobstore datastore."""
 
from azureml.core import Datastore, Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
 
SUBSCRIPTION_ID = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
RESOURCE_GROUP = "buas-y2"
WORKSPACE_NAME = "CV6-2025"
DATASTORE_NAME = "workspaceblobstore"
SRC_DIR = r"C:\Users\vikku\OneDrive\Documenten\Buas\2024-25d-fai2-adsai-ViktoriaKubisova231781\CV_6\data\original_dataset"
TARGET_PATH = "plant_data"
 
def main():
    auth = InteractiveLoginAuthentication()
    workspace = Workspace(
        subscription_id=SUBSCRIPTION_ID,
        resource_group=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
        auth=auth,
    )
    datastore = Datastore(workspace, name=DATASTORE_NAME)
 
    datastore.upload(
        src_dir=SRC_DIR,
        target_path=TARGET_PATH,
        overwrite=True,
        show_progress=True,
    )
 
if __name__ == "__main__":
    main()
 