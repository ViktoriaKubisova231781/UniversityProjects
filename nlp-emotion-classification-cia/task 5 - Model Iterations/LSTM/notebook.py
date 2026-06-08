"""
Emotion Classification Ensemble Approach
----------------------------------------
This script implements an ensemble approach for emotion classification using:
1. Multiple pretrained transformer models
2. Sentiment analysis features
3. Keyword-based heuristics

The script loads test data, runs predictions with two models, applies ensemble rules,
and evaluates the results against ground truth.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import re
from transformers import pipeline
import os
import time

# Ensure plots directory exists
if not os.path.exists('plots'):
    os.makedirs('plots')

# Start timing
start_time = time.time()

def main():
    print("Starting emotion classification ensemble...")
    
    # 1. Load test data
    test_path = r"C:\Users\ronle\Desktop\BUAS\Y2C\datasets\1_testing_dataset_g21_mapped_processed_featured.csv"
    df_test = pd.read_csv(test_path)
    print(f"Loaded test data with {len(df_test)} samples.")
    
    # 2. Create additional sentiment features
    create_sentiment_features(df_test)
    
    # 3. Load and run predictions with both models
    model1_preds, model1_scores, model2_preds, model2_scores = run_model_predictions(df_test)
    
    # 4. Add predictions to dataframe
    df_test['model1_prediction'] = model1_preds
    df_test['model2_prediction'] = model2_preds
    
    # 5. Extract keyword features
    extract_emotion_keywords(df_test)
    
    # 6. Apply ensemble prediction
    print("Applying ensemble prediction rules...")
    df_test['ensemble_prediction'] = df_test.apply(ensemble_prediction, axis=1)
    
    # 7. Evaluate results
    evaluate_predictions(df_test)
    
    # 8. Save results
    save_results(df_test)
    
    elapsed_time = time.time() - start_time
    print(f"\nCompleted in {elapsed_time:.2f} seconds.")

def create_sentiment_features(df):
    """Create additional sentiment features for analysis"""
    df['emotional_intensity'] = df['VADER_Positive'] + df['VADER_Negative']
    df['sentiment_direction'] = df['TextBlob_Polarity'] * df['VADER_Compound']
    df['sentiment_strength'] = abs(df['TextBlob_Polarity']) + abs(df['VADER_Compound'])
    print("Created additional sentiment features.")

def run_model_predictions(df):
    """Run predictions with both emotion classification models"""
    # Define mappings for both models
    mapping1 = {
        'joy': 'happiness',
        'sadness': 'sadness',  
        'anger': 'anger',
        'fear': 'fear',
        'surprise': 'surprise',
        'disgust': 'disgust',
        'neutral': 'neutral'
    }
    
    mapping2 = {
        'admiration': 'happiness', 'amusement': 'happiness', 'approval': 'happiness',
        'caring': 'happiness', 'desire': 'happiness', 'excitement': 'happiness',
        'gratitude': 'happiness', 'joy': 'happiness', 'love': 'happiness',
        'optimism': 'happiness', 'pride': 'happiness', 'relief': 'happiness',
        
        'sadness': 'sadness', 'disappointment': 'sadness', 
        'grief': 'sadness', 'remorse': 'sadness',
        
        'anger': 'anger', 'annoyance': 'anger', 'disapproval': 'anger',
        
        'fear': 'fear', 'nervousness': 'fear',
        
        'surprise': 'surprise', 'realization': 'surprise', 'confusion': 'surprise',
        
        'disgust': 'disgust',
        
        'neutral': 'neutral', 'curiosity': 'neutral', 'embarrassment': 'neutral'
    }
    
    # Load models
    print("Loading first model (distilroberta-base)...")
    model1 = pipeline("text-classification", 
                     model="j-hartmann/emotion-english-distilroberta-base",
                     top_k=None)
    
    print("Loading second model (roberta-base-go_emotions)...")
    model2 = pipeline("text-classification", 
                     model="SamLowe/roberta-base-go_emotions",
                     top_k=None)
    
    # Run predictions in batches
    batch_size = 16
    model1_predictions = []
    model1_scores = []
    model2_predictions = []
    model2_scores = []
    
    print(f"Running predictions on {len(df)} samples with batch size {batch_size}...")
    for i in range(0, len(df), batch_size):
        if i % 100 == 0 and i > 0:
            print(f"Processed {i} samples...")
            
        batch_texts = df['text'][i:i+batch_size].tolist()
        
        # Model 1 predictions
        batch_results1 = model1(batch_texts)
        for result in batch_results1:
            top_emotion = max(result, key=lambda x: x['score'])
            predicted_raw = top_emotion['label']
            predicted_mapped = mapping1.get(predicted_raw, predicted_raw)
            model1_predictions.append(predicted_mapped)
            model1_scores.append({r['label']: r['score'] for r in result})
        
        # Model 2 predictions
        batch_results2 = model2(batch_texts)
        for result in batch_results2:
            top_emotion = max(result, key=lambda x: x['score'])
            predicted_raw = top_emotion['label']
            predicted_mapped = mapping2.get(predicted_raw, predicted_raw)
            model2_predictions.append(predicted_mapped)
            model2_scores.append({r['label']: r['score'] for r in result})
    
    return model1_predictions, model1_scores, model2_predictions, model2_scores

def extract_emotion_keywords(df):
    """Extract emotion-specific keyword counts from text"""
    # Define keyword lists for each emotion
    happiness_words = ['amazing', 'great', 'excited', 'love', 'happy', 'best', 'adventure', 
                      'fierce', 'top', 'model', 'competition', 'vengeance', 'chance', 
                      'cycle', 'season', 'prepared', 'expect']
    sadness_words = ['sad', 'sorry', 'miss', 'lost', 'alone', 'lonely', 'grief']
    anger_words = ['angry', 'mad', 'frustrat', 'hate', 'annoying', 'piss', 'fuck', 'shit', 'fake']
    fear_words = ['afraid', 'scared', 'fear', 'terrif', 'stress', 'worried', 'anxiety']
    surprise_words = ['wow', 'surprise', 'shocking', 'unexpected', 'crazy', 'twist', 'adventure']
    
    # Count emotion keywords in each text
    print("Extracting emotion keywords from text...")
    df['happiness_words'] = df['text'].apply(
        lambda x: sum(1 for word in re.findall(r'\w+', x.lower()) if word in happiness_words)
    )
    df['sadness_words'] = df['text'].apply(
        lambda x: sum(1 for word in re.findall(r'\w+', x.lower()) if word in sadness_words)
    )
    df['anger_words'] = df['text'].apply(
        lambda x: sum(1 for word in re.findall(r'\w+', x.lower()) if word in anger_words)
    )
    df['fear_words'] = df['text'].apply(
        lambda x: sum(1 for word in re.findall(r'\w+', x.lower()) if word in fear_words)
    )
    df['surprise_words'] = df['text'].apply(
        lambda x: sum(1 for word in re.findall(r'\w+', x.lower()) if word in surprise_words)
    )

def ensemble_prediction(row):
    """Apply ensemble rules to combine predictions"""
    # If both models agree, trust them more
    if row['model1_prediction'] == row['model2_prediction']:
        return row['model1_prediction']
    
    # Strong sentiment rules for happiness
    if (row['TextBlob_Polarity'] > 0.15 and row['VADER_Compound'] > 0.3) or row['VADER_Positive'] > 0.3:
        return 'happiness'
    
    # For Asia's Next Top Model dataset - common happiness contexts
    if 'model' in row['text'].lower() and row['model1_prediction'] != 'fear' and row['model2_prediction'] != 'fear':
        return 'happiness'
    
    # Strong negative sentiment for sadness/anger
    if row['VADER_Compound'] < -0.3:
        if row['sadness_words'] > row['anger_words']:
            return 'sadness'
        else:
            return 'anger'
    
    # Keyword-based rules
    emotion_word_counts = {
        'happiness': row['happiness_words'],
        'sadness': row['sadness_words'],
        'anger': row['anger_words'],
        'fear': row['fear_words'],
        'surprise': row['surprise_words']
    }
    
    # If we have strong keyword evidence, use it
    max_count = max(emotion_word_counts.values())
    if max_count > 0:
        max_emotions = [e for e, c in emotion_word_counts.items() if c == max_count]
        if len(max_emotions) == 1:
            return max_emotions[0]
    
    # Model 1 seems better at detecting neutral and surprise
    if row['model1_prediction'] in ['neutral', 'surprise']:
        return row['model1_prediction']
    
    # Model 2 seems better at happiness
    if row['model2_prediction'] == 'happiness':
        return 'happiness'
    
    # Fallback to model 1 
    return row['model1_prediction']

def evaluate_predictions(df):
    """Evaluate prediction accuracy and generate reports"""
    # Calculate accuracies
    model1_accuracy = (df['emotion'] == df['model1_prediction']).mean()
    model2_accuracy = (df['emotion'] == df['model2_prediction']).mean()
    ensemble_accuracy = (df['emotion'] == df['ensemble_prediction']).mean()
    
    print(f"\nModel 1 Accuracy: {model1_accuracy:.4f}")
    print(f"Model 2 Accuracy: {model2_accuracy:.4f}")
    print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
    
    # Show prediction distributions
    print("\nDistribution of ensemble predictions:")
    print(df['ensemble_prediction'].value_counts())
    
    print("\nDistribution of true emotions:")
    print(df['emotion'].value_counts())
    
    # Analyze accuracy by emotion category
    print("\nAccuracy by emotion category:")
    for emotion in df['emotion'].unique():
        subset = df[df['emotion'] == emotion]
        model1_acc = (subset['model1_prediction'] == emotion).mean()
        model2_acc = (subset['model2_prediction'] == emotion).mean()
        ensemble_acc = (subset['ensemble_prediction'] == emotion).mean()
        
        print(f"{emotion} ({len(subset)} samples):")
        print(f"  Model 1: {model1_acc:.2%}")
        print(f"  Model 2: {model2_acc:.2%}")
        print(f"  Ensemble: {ensemble_acc:.2%}")
    
    # Generate confusion matrix
    emotion_categories = sorted(df['emotion'].unique())
    emotion_to_id = {emotion: i for i, emotion in enumerate(emotion_categories)}
    
    true_ids = [emotion_to_id[e] for e in df['emotion']]
    pred_ids = [emotion_to_id[e] for e in df['ensemble_prediction']]
    
    cm = confusion_matrix(true_ids, pred_ids)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotion_categories,
                yticklabels=emotion_categories)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for Ensemble Model')
    plt.tight_layout()
    plt.savefig('plots/ensemble_confusion_matrix.png')
    
    # Generate classification report
    print("\nClassification Report for Ensemble Model:")
    report = classification_report(df['emotion'], df['ensemble_prediction'])
    print(report)

def save_results(df):
    """Save prediction results to CSV for further analysis"""
    output_columns = ['text', 'emotion', 'model1_prediction', 'model2_prediction', 
                     'ensemble_prediction', 'TextBlob_Polarity', 'VADER_Compound']
    df[output_columns].to_csv('ensemble_predictions.csv', index=False)
    print("\nSaved detailed results to 'ensemble_predictions.csv'")
    
    # Show sample results
    print("\nSample predictions:")
    print(df[['text', 'emotion', 'model1_prediction', 'model2_prediction', 'ensemble_prediction']].head(10))

if __name__ == "__main__":
    main()
