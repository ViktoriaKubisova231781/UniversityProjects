"""
Prediction module for emotion classification model.
This module provides functionality for loading trained models and making predictions,
including post-processing to map sub-emotions to main emotions.
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
import glob
from transformers import AutoTokenizer, AutoModel
import io


class BERTClassifier1(nn.Module):
    """
    Multi-task BERT classifier for emotion classification.
    
    This model performs three classification tasks:
    1. Main emotion classification
    2. Sub-emotion classification
    3. Intensity classification
    
    The model combines BERT embeddings with additional features through projection layers.
    """
    
    def __init__(self, model_name, feature_dim, num_classes, hidden_dim=256, dropout=0.1):
        """
        Initialize the BERTClassifier1 model.
        
        Args:
            model_name (str): Name of the pretrained model to use
            feature_dim (int): Dimension of additional features
            num_classes (dict): Dictionary containing number of classes for each task
            hidden_dim (int): Dimension of hidden layers
            dropout (float): Dropout probability
        """
        super().__init__()
        
        # Load base BERT model
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Get BERT embedding dimension
        bert_dim = self.bert.config.hidden_size
        
        # Feature projection layer
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Combine BERT and feature embeddings
        combined_dim = bert_dim + hidden_dim
        
        # Task-specific layers
        self.emotion_classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes['emotion'])
        )
        
        self.sub_emotion_classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes['sub_emotion'])
        )
        
        self.intensity_classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes['intensity'])
        )
    
    def forward(self, input_ids, attention_mask, features):
        """
        Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor): Attention mask
            features (torch.Tensor): Additional features
            
        Returns:
            tuple: (emotion_logits, sub_emotion_logits, intensity_logits)
        """
        # Get BERT embeddings
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_embeddings = bert_output.last_hidden_state[:, 0, :]  # Use [CLS] token
        
        # Project additional features
        projected_features = self.feature_projection(features)
        
        # Combine embeddings
        combined = torch.cat([bert_embeddings, projected_features], dim=1)
        
        # Task-specific predictions
        emotion_logits = self.emotion_classifier(combined)
        sub_emotion_logits = self.sub_emotion_classifier(combined)
        intensity_logits = self.intensity_classifier(combined)
        
        return emotion_logits, sub_emotion_logits, intensity_logits


class ModelLoader:
    """
    A class to handle loading of the model and tokenizer.
    
    This class handles:
    - Loading the BERT model and tokenizer
    - Setting up the device (CPU/GPU)
    - Loading model weights
    """
    
    def __init__(self, model_name="bert-base-multilingual-cased", device=None):
        """
        Initialize the ModelLoader.
        
        Args:
            model_name (str): Name of the pretrained model to use
            device (torch.device, optional): Device to use. If None, will use GPU if available
        """
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        print(f"Loading tokenizer from: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def load_model(self, feature_dim, num_classes, weights_path=None, hidden_dim=256, dropout=0.1):
        """
        Create and load the model.
        
        Args:
            feature_dim (int): Dimension of additional features
            num_classes (dict): Dictionary containing number of classes for each task
            weights_path (str, optional): Path to model weights
            hidden_dim (int): Dimension of hidden layers
            dropout (float): Dropout probability
            
        Returns:
            BERTClassifier1: Loaded model
        """
        # Create model instance
        model = BERTClassifier1(
            model_name=self.model_name,
            feature_dim=feature_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Load weights if provided
        if weights_path is not None:
            print(f"Loading weights from: {weights_path}")
            # Load weights using BytesIO to handle seekable file requirement
            with open(weights_path, 'rb') as f:
                buffer = io.BytesIO(f.read())
                state_dict = torch.load(buffer)
                model.load_state_dict(state_dict)
            print("Successfully loaded model weights")
        
        # Move model to device
        model = model.to(self.device)
        
        return model
    
    def create_predictor(self, model, encoders_dir="./results/encoders"):
        """
        Create a CustomPredictor instance with the loaded model and tokenizer.
        
        Args:
            model (nn.Module): The model instance
            encoders_dir (str): Directory containing the encoder pickle files
            
        Returns:
            CustomPredictor: Predictor instance ready for making predictions
        """
        return CustomPredictor(
            model=model,
            tokenizer=self.tokenizer,
            device=self.device,
            encoders_dir=encoders_dir
        )


class CustomPredictor:
    """
    A class for making predictions with trained emotion classification models.
    
    This class handles:
    - Loading the best model based on test F1 scores
    - Making predictions on new data
    - Post-processing predictions to map sub-emotions to main emotions
    
    Attributes:
        model (nn.Module): The loaded emotion classification model
        tokenizer: Tokenizer for text preprocessing
        device (torch.device): Device to run the model on (CPU/GPU)
        emotion_mapping (dict): Dictionary mapping sub-emotions to main emotions
        encoders (dict): Dictionary of label encoders for each task
    """
    
    def __init__(self, model, tokenizer, device, encoders_dir="./results/encoders"):
        """
        Initialize the EmotionPredictor.
        
        Args:
            model (nn.Module): The base model architecture (unloaded)
            tokenizer: Tokenizer for text preprocessing
            device (torch.device): Device to run the model on
            encoders_dir (str): Directory containing the encoder pickle files
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.feature_dim = model.feature_projection[0].in_features  # Get feature dimension from model
        self.emotion_mapping = {
            'curiosity': "happiness", 
            'neutral': "neutral", 
            'annoyance': "anger", 
            'confusion': "surprise", 
            'disappointment': "sadness",
            'excitement': "happiness", 
            'surprise': "surprise", 
            'realization': "surprise", 
            'desire': "happiness", 
            'amusement': "happiness", 
            'caring': "happiness", 
            'approval': "happiness", 
            'disapproval': "disgust", 
            'nervousness': "fear", 
            'embarrassment': "fear",
            'admiration': "happiness", 
            'pride': "happiness", 
            'anger': "anger",
            'optimism': "happiness", 
            'sadness': "sadness", 
            'joy': "happiness",
            'fear': "fear", 
            'remorse': "sadness",
            'gratitude': "happiness", 
            'disgust': "disgust", 
            'love': "happiness", 
            'relief': "happiness", 
            'grief': "sadness"
        }
        
        # Load encoders
        self.encoders = self._load_encoders(encoders_dir)
        
        # Initialize output tasks
        self.output_tasks = ['emotion', 'sub_emotion', 'intensity']
    
    def _load_encoders(self, encoders_dir):
        """
        Load label encoders from pickle files.
        
        Args:
            encoders_dir (str): Directory containing the encoder pickle files
            
        Returns:
            dict: Dictionary of loaded encoders
        """
        encoders = {}
        for task in ['emotion', 'sub_emotion', 'intensity']:
            with open(f'{encoders_dir}/{task}_encoder.pkl', 'rb') as f:
                encoders[task] = pickle.load(f)
        return encoders
    
    def load_best_model(self, weights_dir="./results/weights", iteration=None, task='sub_emotion'):
        """
        Load the best model based on test F1 scores.
        
        Args:
            weights_dir (str): Directory containing model weights
            iteration (int, optional): Specific iteration to load from
            task (str): Task to optimize for ('emotion', 'sub_emotion', or 'intensity')
            
        Returns:
            float: F1 score of the loaded model
        """
        # Find all model files for the specified task
        pattern = f'best_test_in_{task}_f1_*.pt'
        if iteration is not None:
            pattern = f'best_test_in_{task}_f1_*_iteration_{iteration}.pt'
        
        model_files = glob.glob(os.path.join(weights_dir, pattern))
        
        if not model_files:
            raise FileNotFoundError(f"No model files found matching pattern: {pattern}")
        
        # Find the model with highest F1 score
        best_f1 = 0.0
        best_model_path = None
        
        for model_file in model_files:
            # Extract F1 score from filename
            f1_score = float(model_file.split('f1_')[1].split('_')[0])
            if f1_score > best_f1:
                best_f1 = f1_score
                best_model_path = model_file
        
        print(f"Loading best model from: {os.path.basename(best_model_path)}")
        print(f"Best {task} F1 score: {best_f1:.4f}")
        
        # Load the model weights
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.to(self.device)
        self.model.eval()
        
        return best_f1
    
    def predict(self, texts, batch_size=16):
        """
        Make predictions on new texts.
        
        Args:
            texts (list): List of texts to predict on
            batch_size (int): Batch size for predictions
            
        Returns:
            pd.DataFrame: DataFrame containing predictions and mapped emotions
        """
        # Create a simple dataset for prediction with correct feature dimension
        dataset = PredictionDataset(texts, self.tokenizer, feature_dim=self.feature_dim)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize lists for predictions
        predictions = {task: [] for task in self.output_tasks}
        
        # Generate predictions
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating predictions", ncols=120):
                # Move batch data to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                features = batch['features'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    features=features
                )
                
                # Get predictions for each task
                for i, task in enumerate(self.output_tasks):
                    pred = torch.argmax(outputs[i], dim=1).cpu().numpy()
                    predictions[task].extend(pred)
        
        # Create results DataFrame
        results = pd.DataFrame({'text': texts})
        
        # Convert predictions to original labels
        for task in self.output_tasks:
            results[f'predicted_{task}'] = self.encoders[task].inverse_transform(predictions[task])
        
        # Add mapped emotions
        results = self.post_process(results)
        
        return results
    
    def post_process(self, df):
        """
        Post-process predictions to add mapped emotions.
        
        Args:
            df (pd.DataFrame): DataFrame containing predictions
            
        Returns:
            pd.DataFrame: DataFrame with added mapped emotions
        """
        # Map predicted sub-emotions to main emotions
        df['emotion_pred_post_processed'] = df['predicted_sub_emotion'].map(self.emotion_mapping)
        
        return df


class PredictionDataset(Dataset):
    """Simple dataset for prediction only."""
    
    def __init__(self, texts, tokenizer, feature_dim=0, max_length=128):
        """
        Initialize the dataset.
        
        Args:
            texts (list): List of texts to predict
            tokenizer: Tokenizer for text preprocessing
            feature_dim (int): Dimension of features (should match model's feature_dim)
            max_length (int): Maximum sequence length
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.feature_dim = feature_dim
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Return a dictionary containing the model inputs
        # Note: For prediction, we use zero features matching the model's feature_dim
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'features': torch.zeros(self.feature_dim)  # Use correct feature dimension
        } 