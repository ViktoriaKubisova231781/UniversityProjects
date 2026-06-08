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
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
import traceback
import sys
from data import *

# Suppress specific sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*vocabulary.*")


class BERTClassifier(nn.Module):
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
        Initialize the BERTClassifier model.
        
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
    
    def __init__(self, model_name="microsoft/deberta-v3-base", device=None):
        """
        Initialize the ModelLoader.
        
        Args:
            model_name (str): Name of the pretrained model to use
            device (torch.device, optional): Device to use. If None, will use GPU if available
        """
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # print(f"Using device: {self.device}")
        # print(f"Loading tokenizer from: {model_name}")
        
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
            BERTClassifier: Loaded model
        """
        # Create model instance
        model = BERTClassifier(
            model_name=self.model_name,
            feature_dim=feature_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Load weights if provided
        if weights_path is not None:
            # print(f"Loading weights from: {weights_path}")
            # Load weights using BytesIO to handle seekable file requirement
            with open(weights_path, 'rb') as f:
                buffer = io.BytesIO(f.read())
                state_dict = torch.load(buffer)
                model.load_state_dict(state_dict)
            # print("Successfully loaded model weights")
        
        # Move model to device
        model = model.to(self.device)
        
        return model
    
    def create_predictor(self, model, encoders_dir="./results/encoders", feature_config=None):
        """
        Create a CustomPredictor instance with the loaded model and tokenizer.
        
        Args:
            model (nn.Module): The model instance
            encoders_dir (str): Directory containing the encoder pickle files
            feature_config (dict, optional): Configuration for feature extraction
            
        Returns:
            CustomPredictor: Predictor instance ready for making predictions
        """
        return CustomPredictor(
            model=model,
            tokenizer=self.tokenizer,
            device=self.device,
            encoders_dir=encoders_dir,
            feature_config=feature_config
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
        feature_extractor (FeatureExtractor): Feature extractor for additional features
    """
    
    def __init__(self, model, tokenizer, device, encoders_dir="./results/encoders", feature_config=None):
        """
        Initialize the EmotionPredictor.
        
        Args:
            model (nn.Module): The base model architecture (unloaded)
            tokenizer: Tokenizer for text preprocessing
            device (torch.device): Device to run the model on
            encoders_dir (str): Directory containing the encoder pickle files
            feature_config (dict, optional): Configuration for feature extraction
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        # Get expected feature dimension from model
        self.expected_feature_dim = model.feature_projection[0].in_features
        
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
        
        # Default feature configuration if none is provided
        self.feature_config = feature_config or {
            'pos': False,
            'textblob': False,
            'vader': False,
            'tfidf': False,
            'emolex': False
        }
        
        # Create a feature extractor with the specified configuration
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(current_dir)
        emolex_path = os.path.join(base_dir, "data", "features", "EmoLex", "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")
        self.feature_extractor = FeatureExtractor(feature_config=self.feature_config, lexicon_path=emolex_path)
        
        # Initialize TF-IDF vectorizer with a dummy fit if needed
        # This prevents NotFittedError when calculating feature dimensions
        if self.feature_config.get('tfidf', True):
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.feature_extractor.tfidf_vectorizer = TfidfVectorizer(max_features=100)
            # Fit with a dummy document to initialize the vocabulary
            self.feature_extractor.tfidf_vectorizer.fit(["dummy document"])
        
        # Load encoders
        self.encoders = self._load_encoders(encoders_dir)
        
        # Initialize output tasks
        self.output_tasks = ['emotion', 'sub_emotion', 'intensity']
        
        # Check if feature dimensions will match
        try:
            actual_feature_dim = self.feature_extractor.get_feature_dim()
            if actual_feature_dim != self.expected_feature_dim:
                print(f"WARNING: Feature dimensions mismatch. Model expects {self.expected_feature_dim} features, " 
                      f"but current configuration will produce {actual_feature_dim} features.")
                print("Will use zero padding to match dimensions.")
        except Exception as e:
            print(f"Warning: Could not calculate feature dimensions due to {str(e)}")
            print(f"Will handle dimension matching during prediction.")
    
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
        
        # print(f"Loading best model from: {os.path.basename(best_model_path)}")
        # print(f"Best {task} F1 score: {best_f1:.4f}")
        
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
        # Fit TF-IDF on input texts if needed
        if self.feature_config.get('tfidf', True) and self.feature_extractor.tfidf_vectorizer is not None:
            # print("Fitting TF-IDF vectorizer on input texts...")
            # Make sure we have enough texts to fit, add dummy texts if needed
            fit_texts = texts.copy()
            if len(fit_texts) < 2:
                fit_texts.extend(["dummy document for fitting"]*2)
            self.feature_extractor.fit_tfidf(fit_texts)
        
        # Create dataset with feature extraction and dimension handling
        dataset = EmotionPredictionDataset(
            texts, 
            self.tokenizer, 
            feature_extractor=self.feature_extractor,
            expected_feature_dim=self.expected_feature_dim,
            max_length=128
        )
        
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


class EmotionPredictionDataset(Dataset):
    """Dataset for emotion prediction that handles feature dimension mismatches."""
    
    def __init__(self, texts, tokenizer, feature_extractor=None, expected_feature_dim=0, max_length=128):
        """
        Initialize the dataset.
        
        Args:
            texts (list): List of texts to predict
            tokenizer: Tokenizer for text preprocessing
            feature_extractor (FeatureExtractor, optional): Feature extractor instance
            expected_feature_dim (int): Expected feature dimension for the model
            max_length (int): Maximum sequence length
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.feature_extractor = feature_extractor
        self.expected_feature_dim = expected_feature_dim
    
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
        
        # Extract features if feature extractor is provided
        if self.feature_extractor is not None:
            try:
                features = self.feature_extractor.extract_all_features(text)
                actual_dim = len(features)
                
                # Handle dimension mismatch
                if actual_dim < self.expected_feature_dim:
                    # Pad with zeros if actual dimension is smaller
                    padded_features = np.zeros(self.expected_feature_dim)
                    padded_features[:actual_dim] = features
                    features_tensor = torch.tensor(padded_features, dtype=torch.float32)
                elif actual_dim > self.expected_feature_dim:
                    # Truncate if actual dimension is larger
                    features_tensor = torch.tensor(features[:self.expected_feature_dim], dtype=torch.float32)
                else:
                    features_tensor = torch.tensor(features, dtype=torch.float32)
            except Exception as e:
                # Fallback to zeros if feature extraction fails
                print(f"Warning: Feature extraction failed for text: '{text[:30]}...' - {str(e)}")
                features_tensor = torch.zeros(self.expected_feature_dim)
        else:
            # Use zero tensor if no feature extractor
            features_tensor = torch.zeros(self.expected_feature_dim)
        
        # Return a dictionary containing the model inputs
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'features': features_tensor
        }


class PredictionDataset(Dataset):
    """Simple dataset for prediction only (for backward compatibility)."""
    
    def __init__(self, texts, tokenizer, feature_extractor=None, feature_config=None, max_length=128):
        """
        Initialize the dataset.
        
        Args:
            texts (list): List of texts to predict
            tokenizer: Tokenizer for text preprocessing
            feature_extractor (FeatureExtractor, optional): Feature extractor instance
            feature_config (dict, optional): Configuration for feature extraction
            max_length (int): Maximum sequence length
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.feature_extractor = feature_extractor
        self.feature_config = feature_config
        
        # If feature_extractor is not provided, determine feature dimension
        # from the feature_config or default to 0
        if self.feature_extractor is None:
            # Default feature dimension
            self.feature_dim = 0
        else:
            # Use the feature extractor's dimension
            self.feature_dim = self.feature_extractor.get_feature_dim()
    
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
        
        # If feature extractor is provided, extract features
        # Otherwise use zero features
        if self.feature_extractor is not None:
            features = self.feature_extractor.extract_all_features(text)
            features_tensor = torch.tensor(features, dtype=torch.float32)
        else:
            features_tensor = torch.zeros(self.feature_dim)
        
        # Return a dictionary containing the model inputs
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'features': features_tensor
        }

class EmotionPredictor:
    def __init__(self):
        self._model = None
        self._predictor = None
    
    def predict(self, texts, feature_config=None, reload_model=False):
        """
        Predicts emotions for the given text(s)
        
        Args:
            texts (str or list): Text or list of texts to analyze
            feature_config (dict, optional): Configuration for features to use in prediction
            reload_model (bool): Force reload the model even if cached
            
        Returns:
            dict or list: Dictionary with emotion predictions for a single text or
                        list of dictionaries for multiple texts
        """
        # Convert single text to list for uniform processing
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
            
        # Default feature configuration if not provided
        if feature_config is None:
            feature_config = {
                'pos': False,
                'textblob': False,
                'vader': False,
                'tfidf': True,
                'emolex': True
            }
        
        # Load the model and create predictor if not already loaded or if reload is requested
        if self._model is None or self._predictor is None or reload_model:
            # Get absolute path to base directory (parent of src)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(current_dir)

            # Initialize model loader
            loader = ModelLoader("microsoft/deberta-v3-xsmall")
            
            # Tokenizer
            tokenizer = loader.tokenizer    
            feature_dim = 121

            # Load model
            num_classes = {
                'emotion': 7,
                'sub_emotion': 28,
                'intensity': 3
            }

            model_path = os.path.join(base_dir, "models", "best_test_in_emotion_f1_0.7851_iteration_5.pt")
            self._model = loader.load_model(
                feature_dim=feature_dim,
                num_classes=num_classes,
                weights_path=model_path 
            )

            # Create predictor with feature configuration
            encoders_dir = os.path.join(base_dir, "data", "encoders")
            self._predictor = loader.create_predictor(
                model=self._model,
                encoders_dir=encoders_dir,
                feature_config=feature_config
            )
        
        # Make predictions
        results = self._predictor.predict(texts)
        
        # Format the output
        output = []
        for i, text in enumerate(texts):
            prediction = {
                'text': text,
                'emotion': results.loc[i, "predicted_emotion"],
                'sub_emotion': results.loc[i, "predicted_sub_emotion"],
                'intensity': results.loc[i, "predicted_intensity"]
            }
            output.append(prediction)
        
        # Return single dictionary if input was a single text
        if single_input:
            return output[0]
        
        return output