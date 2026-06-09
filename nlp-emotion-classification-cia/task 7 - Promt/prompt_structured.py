import requests
import json
import pandas as pd
import time
import os
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report

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
output_filename = "test_structured_prompt.csv"
output_path = os.path.join(output_dir, output_filename)

# Read the CSV file
df = pd.read_csv(csv_path)

# Create columns for the emotion result and emotion code if they don't exist
if 'emotion' not in df.columns:
    df['emotion'] = ""
if 'emotion_code' not in df.columns:
    df['emotion_code'] = -1

# Store the original emotion values for later comparison
original_emotions = df['emotion'].copy()
original_emotion_codes = df['emotion_code'].copy()

# Function to analyze emotions in text with structured prompt
def analyze_emotion(text):
    # Structured prompt with formatting instructions
    prompt = """TASK: Classify the emotional content of the text below into EXACTLY ONE of the following categories: happiness, sadness, anger, fear, surprise, disgust, neutral.

TEXT: "{}"

INSTRUCTIONS:
1. Read the text carefully
2. Identify the primary emotion expressed
3. Choose ONLY ONE emotion from: happiness, sadness, anger, fear, surprise, disgust, neutral
4. Format your response using only the template below

RESPONSE TEMPLATE:
EMOTION: [emotion]

DO NOT include any explanations or additional text. Only provide the emotion name in lowercase using the exact template above.""".format(text)
    
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
        response_text = response_data['candidates'][0]['content']['parts'][0]['text'].strip().lower()
        
        # Look for a structured response like "EMOTION: happiness"
        if "emotion:" in response_text:
            emotion = response_text.split("emotion:")[1].strip()
            if emotion in VALID_EMOTIONS:
                return emotion
        
        # If structured extraction fails, try to find any valid emotion in the response
        for valid_emotion in VALID_EMOTIONS:
            if valid_emotion in response_text:
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
    df.at[index, 'emotion'] = emotion
    df.at[index, 'emotion_code'] = emotion_mapping.get(emotion, -1)
    
    # Save intermediate results
    if index % 10 == 0 and index > 0:
        df.to_csv(output_path, index=False)
        print(f"Saved progress after {index} entries")
    
    # Avoid rate limiting
    time.sleep(1)

# Save the final results
df.to_csv(output_path, index=False)

print(f"\nEmotion analysis completed and saved to {output_path}")

# Add columns for comparison if original emotion data exists
if 'emotion' in df.columns and len(original_emotions) == len(df):
    df['original_emotion'] = original_emotions
    df['original_emotion_code'] = original_emotion_codes
    
    # Calculate accuracy
    matching = (df['emotion_code'] == df['original_emotion_code']).sum()
    accuracy = matching / len(df) * 100
    
    print(f"\nAccuracy compared to original labels: {accuracy:.2f}%")
    
    # Calculate F1 score
    try:
        f1 = f1_score(df['original_emotion_code'], df['emotion_code'], average='weighted')
        print(f"Weighted F1 Score: {f1:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(df['original_emotion_code'], df['emotion_code']))
    except Exception as e:
        print(f"Error calculating F1 score: {e}")
    
    # Save the updated dataframe with comparison data
    df.to_csv(output_path, index=False) 