import requests
import json
import pandas as pd
import time
import os
from sklearn.metrics import f1_score, classification_report

# Set the API key directly
GEMINI_API_KEY = "#"

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
VALID_EMOTIONS = ['happiness', 'sadness', 'anger', 'surprise', 'fear', 'disgust', 'neutral']

# Emotion code mapping
emotion_mapping = {
    'happiness': 0,
    'sadness': 1,
    'anger': 2,
    'fear': 3,
    'surprise': 4,
    'disgust': 5,
    'neutral': 6
}

# Path to the CSV file
csv_path = r"c:\Users\ronle\Desktop\BUAS\2024-25c-fai2-adsai-group-group21\Task 7 - Promt\dataset\test_g21.csv"

# Output path
output_dir = r"c:\Users\ronle\Desktop\BUAS\2024-25c-fai2-adsai-group-group21\Task 7 - Promt\dataset"
output_filename = "test_baseline_prompt.csv"
output_path = os.path.join(output_dir, output_filename)

# Read the CSV file
df = pd.read_csv(csv_path)

# Store the original emotion values
original_emotions = df['emotion'].copy()
original_emotion_codes = df['emotion_code'].copy()

# Function to analyze emotions in text with baseline prompt
def analyze_emotion(text):
    # Baseline prompt - simple and direct
    prompt = f"Analyze the following sentence and classify it as one of the six core emotions (happiness, sadness, anger, surprise, fear, disgust) or neutral: {text}"
    
    # Request body
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    
    # Make the request
    response = requests.post(url, params=params, headers=headers, json=data)
    
    # Get the response data
    response_data = response.json()
    
    try:
        # Extract the emotion analysis
        emotion = response_data['candidates'][0]['content']['parts'][0]['text'].strip().lower()
        
        # Find the closest match to one of our valid emotions
        for valid_emotion in VALID_EMOTIONS:
            if valid_emotion in emotion:
                return valid_emotion
        
        # Default if no match found
        return "neutral"
    except (KeyError, IndexError):
        return "neutral"  # Default if error

# Process each row in the dataframe
for index, row in df.iterrows():
    if pd.isna(row['cleaned_text']) or not row['cleaned_text'].strip():
        continue  # Skip empty text
        
    print(f"Analyzing text {index+1}/{len(df)}: {row['cleaned_text'][:50]}...")
    
    # Analyze the text
    emotion = analyze_emotion(row['cleaned_text'])
    
    # Update the dataframe with emotion and its code
    df.at[index, 'predicted_emotion'] = emotion
    df.at[index, 'predicted_emotion_code'] = emotion_mapping.get(emotion, -1)
    
    # Save intermediate results
    if index % 10 == 0 and index > 0:
        df.to_csv(output_path, index=False)
        print(f"Saved progress after {index} entries")
    
    # Avoid rate limiting
    time.sleep(1)

# Save the final results
df.to_csv(output_path, index=False)

print(f"\nEmotion analysis completed and saved to {output_path}")

# Calculate accuracy and F1 score
valid_indices = (df['emotion_code'] != -1) & (df['predicted_emotion_code'] != -1)
true_labels = df.loc[valid_indices, 'emotion_code']
pred_labels = df.loc[valid_indices, 'predicted_emotion_code']

if len(true_labels) > 0:
    # Calculate accuracy
    matching = (pred_labels == true_labels).sum()
    accuracy = matching / len(true_labels) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")
    
    # Calculate F1 score
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    print(f"Weighted F1 Score: {f1:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels)) 