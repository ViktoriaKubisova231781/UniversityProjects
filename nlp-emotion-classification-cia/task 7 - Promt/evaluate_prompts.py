import pandas as pd
import numpy as np
import os
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths
base_dir = r"c:\Users\ronle\Desktop\BUAS\2024-25c-fai2-adsai-group-group21"
dataset_dir = os.path.join(base_dir, "Task 7 - Promt", "dataset")
original_dataset_path = os.path.join(dataset_dir, "test_g21.csv")

# Define generated datasets to compare
datasets = {
    "Baseline": "test_baseline_prompt.csv",
    "Few-Shot": "test_few_shot_prompt.csv",
    "Definitions": "test_definitions_prompt.csv",
    "Combined": "test_combined_prompt.csv",
    "Structured": "test_structured_prompt.csv"
}

# Load the original dataset to get ground truth
try:
    original_df = pd.read_csv(original_dataset_path)
    print(f"Successfully loaded original dataset from {original_dataset_path}")
    
    # Check if emotion_code exists
    if 'emotion_code' in original_df.columns:
        print(f"Found {len(original_df)} rows with emotion_code column")
        
        # Create a mapping from index to original emotion_code
        original_emotions = {}
        for idx, row in original_df.iterrows():
            original_emotions[idx] = row['emotion_code']
        
        print(f"Extracted {len(original_emotions)} original emotion codes")
    else:
        print("Original dataset does not contain emotion_code column")
        original_emotions = {}
except Exception as e:
    print(f"Error loading original dataset: {e}")
    original_emotions = {}

# Function to load dataset
def load_dataset(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

# Function to create and display confusion matrix
def plot_confusion_matrix(cm, classes, title, filename):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_dir, filename))
    plt.close()

# Dictionary to store results
results = {}

# Process each dataset
for name, filename in datasets.items():
    file_path = os.path.join(dataset_dir, filename)
    df = load_dataset(file_path)
    
    if df is None:
        print(f"Skipping {name} dataset due to loading error")
        continue
    
    if 'emotion_code' not in df.columns:
        print(f"Dataset {name} does not contain emotion_code column")
        continue
    
    # Get predicted and true labels
    true_labels = []
    pred_labels = []
    
    for idx, row in df.iterrows():
        if idx in original_emotions:
            true_labels.append(original_emotions[idx])
            pred_labels.append(row['emotion_code'])
    
    if len(true_labels) > 0:
        # Calculate F1 score
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)
        
        # Remove entries where either true or predicted is -1 (unknown)
        valid_indices = (true_labels != -1) & (pred_labels != -1)
        true_labels = true_labels[valid_indices]
        pred_labels = pred_labels[valid_indices]
        
        if len(true_labels) > 0:
            f1 = f1_score(true_labels, pred_labels, average='weighted')
            results[name] = {
                'f1_score': f1,
                'cm': confusion_matrix(true_labels, pred_labels)
            }
            
            # Print results
            print(f"\n--- Results for {name} Prompt ---")
            print(f"Weighted F1 Score: {f1:.4f}")
            print("\nClassification Report:")
            print(classification_report(true_labels, pred_labels))
            
            # Generate confusion matrix
            emotion_names = ['Happiness', 'Sadness', 'Anger', 'Fear', 'Surprise', 'Disgust', 'Neutral']
            plot_confusion_matrix(
                results[name]['cm'], 
                emotion_names, 
                f'Confusion Matrix - {name} Prompt', 
                f'cm_{name.lower()}.png'
            )
        else:
            print(f"No valid entries found for {name} dataset")
    else:
        print(f"No matching indices found between {name} dataset and original dataset")
        
# Compare all results
if results:
    # Create summary dataframe
    summary = pd.DataFrame({
        'Prompt Type': list(results.keys()),
        'F1 Score': [results[k]['f1_score'] for k in results]
    })
    
    # Sort by F1 score
    summary = summary.sort_values('F1 Score', ascending=False).reset_index(drop=True)
    
    print("\n=== SUMMARY OF RESULTS ===")
    print(summary)
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(summary['Prompt Type'], summary['F1 Score'], color='skyblue')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')
    
    plt.title('F1 Score Comparison Across Different Prompt Types')
    plt.xlabel('Prompt Type')
    plt.ylabel('Weighted F1 Score')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_dir, 'prompt_comparison.png'))
    
    # Create a report for the best performing prompt
    best_prompt = summary.iloc[0]['Prompt Type']
    best_f1 = summary.iloc[0]['F1 Score']
    
    with open(os.path.join(dataset_dir, 'prompt_engineering_log.md'), 'w') as f:
        f.write("# Prompt Engineering Log\n\n")
        f.write("## Summary\n\n")
        f.write(f"Best performing prompt: **{best_prompt}** with F1 Score of **{best_f1:.4f}**\n\n")
        f.write("### F1 Scores for all prompts:\n\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n\n## Prompt Descriptions\n\n")
        
        f.write("### Baseline Prompt\n")
        f.write("A simple direct prompt asking to classify the emotion with minimal instructions.\n")
        f.write("```\nAnalyze the following sentence and classify it as one of the six core emotions (happiness, sadness, anger, surprise, fear, disgust) or neutral: [text]\n```\n\n")
        
        f.write("### Few-Shot Prompt\n")
        f.write("Includes multiple examples of sentences and their corresponding emotion labels to guide the model.\n")
        f.write("```\nClassify the emotion in the following text as one of these emotions: happiness, sadness, anger, surprise, fear, disgust, neutral.\n\nHere are some examples:\n1. \"I'm so excited about my new job!\" -> happiness\n...[more examples]\n\nNow classify this text: \"[text]\"\n```\n\n")
        
        f.write("### Definitions Prompt\n")
        f.write("Provides detailed definitions of each emotion category to help the model better understand the distinctions.\n")
        f.write("```\nAnalyze the following sentence and classify its emotional content as one of these emotions: \n\n1. Happiness: A feeling of pleasure, contentment, or joy...\n...[definitions for each emotion]\n\nText: \"[text]\"\n```\n\n")
        
        f.write("### Combined Prompt\n")
        f.write("Combines both detailed definitions and examples for each emotion category, plus clear instructions on formatting the response.\n")
        f.write("```\nI need you to analyze the emotion expressed in a text. Classify it as exactly one of these emotions: happiness, sadness, anger, fear, surprise, disgust, or neutral.\n\nEMOTION DEFINITIONS:\n[definitions]\n\nEXAMPLES:\n[examples]\n\nIMPORTANT: Reply with only ONE word - the emotion name.\n\nTEXT TO ANALYZE: \"[text]\"\n```\n\n")
        
        f.write("### Structured Prompt\n")
        f.write("Focuses on strict response formatting with clear instructions on how to provide the answer in a consistent format.\n")
        f.write("```\nTASK: Classify the emotional content of the text below into EXACTLY ONE of the following categories: happiness, sadness, anger, fear, surprise, disgust, neutral.\n\nTEXT: \"[text]\"\n\nINSTRUCTIONS:\n1. Read the text carefully\n2. Identify the primary emotion expressed\n3. Choose ONLY ONE emotion from: happiness, sadness, anger, fear, surprise, disgust, neutral\n4. Format your response using only the template below\n\nRESPONSE TEMPLATE:\nEMOTION: [emotion]\n```\n\n")
        
        f.write("\n## Conclusion\n\n")
        f.write(f"The {best_prompt} prompt achieved the best performance with an F1 score of {best_f1:.4f}. ")
        
        if best_prompt == "Combined":
            f.write("This suggests that providing both clear definitions and examples helps the model better understand the nuanced differences between emotion categories.")
        elif best_prompt == "Few-Shot":
            f.write("This suggests that concrete examples are more valuable than abstract definitions for this task.")
        elif best_prompt == "Definitions":
            f.write("This suggests that detailed emotion definitions are crucial for accurate classification.")
        elif best_prompt == "Structured":
            f.write("This suggests that enforcing a strict response format helps the model focus on delivering consistent and accurate classifications.")
        else:
            f.write("This suggests that simpler prompts may be more effective for this particular task.")
    
    print(f"\nPrompt engineering log saved to {os.path.join(dataset_dir, 'prompt_engineering_log.md')}")
else:
    print("No results to compare. Please check if datasets contain the required emotion code columns.") 