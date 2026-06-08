from azure.identity import InteractiveBrowserCredential
from azure.ai.ml import MLClient, Input, Output
from azure.ai.ml.entities import CommandJob
from azure.ai.ml.entities._job.command import CommandJobLimits

# Connect to your Azure ML workspace
ml_client = MLClient(
    credential=InteractiveBrowserCredential(),
    subscription_id="0a94de80-6d3b-49f2-b3e9-ec5818862801",
    resource_group_name="buas-y2",
    workspace_name="CV6-2025",
)

# Reference raw folders from the default datastore path
image_input = Input(
    type="uri_folder",
    path="azureml://datastores/workspaceblobstore/paths/plant_data/images"
)

mask_input = Input(
    type="uri_folder",
    path="azureml://datastores/workspaceblobstore/paths/plant_data/masks"
)

# Define the job
job = CommandJob(
    code="./azureml",
    command=(
        "python preprocess_entry.py "
        "--image_dir ${{inputs.image_dir}} "
        "--mask_dir ${{inputs.mask_dir}} "
        "--train_output_dir ${{outputs.train_output}} "
        "--val_output_dir ${{outputs.val_output}} "
        "--test_output_dir ${{outputs.test_output}}"
    ),
    environment="iris@latest",
    compute="adsai-lambda-0",
    inputs={
        "image_dir": image_input,
        "mask_dir": mask_input
    },
    outputs={
        "train_output": Output(type="uri_folder", mode="upload"),
        "val_output": Output(type="uri_folder", mode="upload"),
        "test_output": Output(type="uri_folder", mode="upload")
    },
    display_name="preprocess-from-blob",
    experiment_name="preprocessing_test",
    limits=CommandJobLimits(timeout=3600)
)

# Submit to Azure ML
returned_job = ml_client.jobs.create_or_update(job)
print(f"Submitted job: {returned_job.name}")
print(f"Monitor at: {returned_job.studio_url}")
