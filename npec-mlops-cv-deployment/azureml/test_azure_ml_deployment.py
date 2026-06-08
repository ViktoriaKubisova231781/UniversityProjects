import requests
import json

url = "https://cv6-2025-viki-unet-model.westeurope.inference.ml.azure.com/score"
api_key = "<YOUR_API_KEY>"  # 🔐 Replace this with your real key from Azure ML

# Replace this with the name of an actual image to test
with open("C:\\Users\\vikku\\OneDrive\\Documenten\\Buas\\2024-25d-fai2-adsai-ViktoriaKubisova231781\\sample_dataset\\images\\train_Alican_212231_im3.png", "rb") as f:
    image_bytes = f.read()

payload = {
    "image": image_bytes.decode("latin1")  # Match with your score.py encoding
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

response = requests.post(url, headers=headers, data=json.dumps(payload))
print(response.json())
