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
output_filename = "test_combined_prompt.csv"
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

# Function to analyze emotions in text with combined approach
def analyze_emotion(text):
    # Combined prompt with detailed definitions and examples
    prompt = """I need you to analyze the emotion expressed in a text. Classify it as exactly one of these emotions: happiness, sadness, anger, fear, surprise, disgust, or neutral.

EMOTION DEFINITIONS:
1. Happiness: A feeling of pleasure, contentment, or joy. Positive emotions like cheerfulness, delight, gratitude, optimism, or satisfaction.
2. Sadness: A feeling of unhappiness, grief, or sorrow. Emotions like disappointment, despair, gloom, loneliness, or regret.
3. Anger: A strong feeling of annoyance, displeasure, or hostility. Emotions like frustration, irritation, rage, resentment, or contempt.
4. Fear: An unpleasant emotion caused by the threat of danger, pain, or harm. Emotions like anxiety, dread, nervousness, terror, or worry.
5. Surprise: A feeling of astonishment or shock caused by something unexpected. Emotions like amazement, bewilderment, or disbelief.
6. Disgust: A feeling of revulsion or strong disapproval. Emotions like aversion, distaste, repulsion, or loathing.
7. Neutral: Absence of any strong or obvious emotion. The text states facts, observations, or thoughts without emotional content.

EXAMPLES:
- "I'm so excited about my new job!" - happiness
- "I can't believe she's gone forever." - sadness
- "I'm furious that they lied to me again!" - anger
- "I'm terrified of what might happen next." - fear
- "I wasn't expecting to see you here!" - surprise
- "That smell makes me feel sick." - disgust
- "I'm going to the store to buy some groceries." - neutral
- "I love spending time with you." - happiness
- "I failed my exam even though I studied hard." - sadness

IMPORTANT: Reply with only ONE word - the emotion name (happiness, sadness, anger, fear, surprise, disgust, or neutral).

TEXT TO ANALYZE: "{}"

EMOTION:""".format(text)
    
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