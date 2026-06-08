import requests
import json
import pandas as pd
import time
import os

# Set the API key directly
GEMINI_API_KEY = "AIzaSyDbX7aF9a692IDbWsTKS_WSVBzH9WJVK6A"

# API endpoint
url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Parameters including the API key
params = {
    "key": GEMINI_API_KEY
}

# Headers
headers = {
    "Content-Type": "application/json"
}

# Define the valid emotions
VALID_EMOTIONS = ['surprise', 'neutral', 'anger', 'sadness', 'happiness', 'fear', 'disgust']

# Path to the CSV file
csv_path = r"C:\Users\ronle\Desktop\BUAS\2024-25c-fai2-adsai-group-group21\data\raw\test.csv"

# Read the CSV file
df = pd.read_csv(csv_path)

# Create column for the emotion result if it doesn't exist
if 'emotion' not in df.columns:
    df['emotion'] = ""

# Function to analyze emotions in text
def analyze_emotion(text):
    # Request body
    data = {
        "contents": [{
            "parts": [{"text": f"Analyze the emotion in this text and respond with exactly one emotion from this list: surprise, neutral, anger, sadness, happiness, fear, disgust. Only respond with one of these words and nothing else: \n\n{text}"}]
        }]
    }
    
    # Make the request
    response = requests.post(url, params=params, headers=headers, json=data)
    
    # Get the response data
    response_data = response.json()
    
    try:
        # Extract the emotion analysis
        emotion = response_data['candidates'][0]['content']['parts'][0]['text'].strip().lower()
        
        # Ensure it's one of our valid emotions
        if emotion in VALID_EMOTIONS:
            return emotion
        else:
            # Find the closest match
            for valid_emotion in VALID_EMOTIONS:
                if valid_emotion in emotion:
                    return valid_emotion
            return "neutral"  # Default if no match found
    except (KeyError, IndexError):
        return "neutral"  # Default if error

# Process each row in the dataframe
for index, row in df.iterrows():
    if pd.isna(row['text']) or not row['text'].strip():
        continue  # Skip empty text
        
    print(f"Analyzing text {index+1}/{len(df)}: {row['text'][:50]}...")
    
    # Analyze the text
    emotion = analyze_emotion(row['text'])
    
    # Update the dataframe
    df.at[index, 'emotion'] = emotion
    
    # Save intermediate results
    if index % 10 == 0 and index > 0:
        df.to_csv(csv_path, index=False)
        print(f"Saved progress after {index} entries")
    
    # Avoid rate limiting
    time.sleep(1)

# Save the final results
output_path = os.path.join(os.path.dirname(csv_path), "test_with_emotions.csv")
df.to_csv(output_path, index=False)

print(f"\nEmotion analysis completed and saved to {output_path}")