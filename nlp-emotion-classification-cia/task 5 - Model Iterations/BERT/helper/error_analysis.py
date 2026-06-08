"""
Error Analysis Module for Emotion Classification Model

This module provides comprehensive tools for analyzing and visualizing errors in emotion classification models.
It includes functionality for error pattern analysis, visualization, and generating improvement recommendations.

Classes:
    AnalysisConfig: Configuration dataclass for error analysis visualization and reporting
    ErrorAnalysis: Main class for performing error analysis on emotion classification models

Dependencies:
    - torch: PyTorch for deep learning operations
    - numpy: Numerical computing
    - pandas: Data manipulation and analysis
    - matplotlib: Basic plotting
    - seaborn: Statistical visualization
    - sklearn: Machine learning utilities
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import json
import os
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

@dataclass
class AnalysisConfig:
    """
    Configuration dataclass for error analysis visualization and reporting.
    
    This class provides customizable parameters for controlling the appearance and behavior
    of error analysis visualizations and reports.
    
    Attributes:
        figure_size (Tuple[int, int]): Size of the output figures in inches (width, height)
        font_size (Dict[str, int]): Font sizes for different text elements in plots
        color_palette (str): Name of the seaborn color palette to use
        max_text_length (int): Maximum length of text to display in examples
        example_count (int): Number of example errors to display
        correlation_threshold (float): Threshold for considering correlations significant
        save_plots (bool): Whether to save generated plots to files
        plot_format (str): Format to save plots in (e.g., 'png', 'pdf')
        output_dir (str): Directory to save analysis results and plots
    """
    figure_size: Tuple[int, int] = (12, 6)
    font_size: Dict[str, int] = None
    color_palette: str = "viridis"
    max_text_length: int = 70
    example_count: int = 30
    correlation_threshold: float = 0.1
    save_plots: bool = True
    plot_format: str = "png"
    output_dir: str = "./results/error_analysis"

    def __post_init__(self):
        """
        Initialize default font sizes if not provided.
        
        Sets up a default dictionary of font sizes for different plot elements
        if no custom font sizes are specified.
        """
        if self.font_size is None:
            self.font_size = {
                'title': 16,
                'label': 12,
                'tick': 10,
                'annotation': 10
            }

class ErrorAnalysis:
    """
    A comprehensive error analysis tool for emotion classification models.
    
    This class provides methods for analyzing model errors, visualizing error patterns,
    and generating recommendations for model improvement. It supports multi-task emotion
    classification with emotion, sub-emotion, and intensity prediction.
    
    Attributes:
        model (torch.nn.Module): The trained emotion classification model
        test_dataloader (torch.utils.data.DataLoader): DataLoader for test data
        device (torch.device): Device to run the model on (CPU/GPU)
        train_df (pd.DataFrame): Training dataset for fitting label encoders
        test_df (pd.DataFrame): Test dataset for analysis
        config (AnalysisConfig): Configuration for analysis and visualization
        label_encoders (Dict[str, LabelEncoder]): Label encoders for each task
        results_df (pd.DataFrame): Complete results of the analysis
        correct_df (pd.DataFrame): Correctly classified examples
        error_df (pd.DataFrame): Misclassified examples
        error_analysis (Dict): Detailed error analysis results
        performance_metrics (Dict): Model performance metrics
        timestamp (str): Timestamp of the analysis
    """
    
    def __init__(self, 
                 model: torch.nn.Module,
                 test_dataloader: torch.utils.data.DataLoader,
                 device: torch.device,
                 train_df: pd.DataFrame,
                 test_df: pd.DataFrame,
                 config: Optional[AnalysisConfig] = None):
        """
        Initialize the ErrorAnalysis class.
        
        Args:
            model (torch.nn.Module): The trained model to analyze
            test_dataloader (torch.utils.data.DataLoader): DataLoader containing test data
            device (torch.device): Device to run the model on (CPU/GPU)
            train_df (pd.DataFrame): Training dataset for fitting label encoders
            test_df (pd.DataFrame): Test dataset for analysis
            config (Optional[AnalysisConfig]): Configuration for analysis and visualization.
                                             If None, uses default configuration.
        
        Note:
            The initialization process:
            1. Sets up model and data attributes
            2. Initializes and fits label encoders on training data
            3. Creates output directory for results
            4. Sets up visualization style
        """
        # Store model and data attributes
        self.model = model
        self.test_dataloader = test_dataloader
        self.device = device
        self.train_df = train_df
        self.test_df = test_df
        self.config = config or AnalysisConfig()
        
        # Initialize label encoders for each task
        self.label_encoders = {
            'emotion': LabelEncoder(),
            'sub_emotion': LabelEncoder(),
            'intensity': LabelEncoder()
        }
        
        # Fit label encoders on training data and transform both datasets
        for col in ['emotion', 'sub_emotion', 'intensity']:
            self.label_encoders[col].fit(train_df[col])
            train_df[f'{col}_encoded'] = self.label_encoders[col].transform(train_df[col])
            
            if test_df is not None:
                test_df[f'{col}_encoded'] = self.label_encoders[col].transform(test_df[col])
        
        # Initialize state variables
        self.results_df = None
        self.correct_df = None
        self.error_df = None
        self.error_analysis = None
        self.performance_metrics = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Set up visualization style
        self._setup_visualization_style()

    def _setup_visualization_style(self):
        """
        Set up consistent visualization style for all plots.
        
        Configures seaborn and matplotlib settings to ensure consistent
        and professional-looking visualizations across all analysis plots.
        """
        # Set seaborn color palette
        sns.set_palette(self.config.color_palette)
        
        # Configure matplotlib default font sizes
        plt.rcParams['font.size'] = self.config.font_size['tick']
        plt.rcParams['axes.titlesize'] = self.config.font_size['title']
        plt.rcParams['axes.labelsize'] = self.config.font_size['label']

    def analyze_errors(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Analyze model errors to identify patterns and improvement areas.
        
        This method performs a comprehensive analysis of model predictions on the test set,
        including:
        1. Making predictions on all test examples
        2. Computing prediction probabilities
        3. Comparing predictions with true labels
        4. Calculating various correctness metrics
        5. Separating correct and incorrect predictions
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing:
                - results_df: Complete results with all predictions and metrics
                - correct_df: Subset of correctly classified examples
                - error_df: Subset of misclassified examples
        
        Note:
            The method sets the model to evaluation mode and uses torch.no_grad()
            for efficient inference. It processes the test set in batches and
            combines results for comprehensive analysis.
        """
        # Set model to evaluation mode for inference
        self.model.eval()
        
        # Initialize lists to store results
        predictions = []
        true_labels = []
        texts = []
        probabilities = []
        
        # Process test set in batches
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.test_dataloader, desc="Predicting on test set")):
                # Move batch data to appropriate device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                features = batch['features'].to(self.device)
                
                # Get corresponding texts from original dataset
                batch_start = i * self.test_dataloader.batch_size
                batch_end = min((i + 1) * self.test_dataloader.batch_size, len(self.test_df))
                batch_texts = self.test_df['text'].iloc[batch_start:batch_end].tolist()
                texts.extend(batch_texts)
                
                # Forward pass through model
                emotion_logits, sub_emotion_logits, intensity_logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    features=features
                )
                
                # Get predictions and probabilities for each task
                preds = {
                    'emotion': torch.argmax(emotion_logits, dim=1).cpu().numpy(),
                    'sub_emotion': torch.argmax(sub_emotion_logits, dim=1).cpu().numpy(),
                    'intensity': torch.argmax(intensity_logits, dim=1).cpu().numpy()
                }
                    
                probs = {
                    'emotion': torch.softmax(emotion_logits, dim=1).cpu().numpy(),
                    'sub_emotion': torch.softmax(sub_emotion_logits, dim=1).cpu().numpy(),
                    'intensity': torch.softmax(intensity_logits, dim=1).cpu().numpy()
                }
                
                # Get true labels from batch
                labels = {
                    'emotion': batch['emotion_label'].cpu().numpy(),
                    'sub_emotion': batch['sub_emotion_label'].cpu().numpy(),
                    'intensity': batch['intensity_label'].cpu().numpy()
                }
                
                # Store results for this batch
                predictions.append(preds)
                true_labels.append(labels)
                probabilities.append(probs)
    
        # Combine results from all batches
        all_preds = {
            'emotion': np.concatenate([p['emotion'] for p in predictions]),
            'sub_emotion': np.concatenate([p['sub_emotion'] for p in predictions]),
            'intensity': np.concatenate([p['intensity'] for p in predictions])
        }
        
        all_labels = {
            'emotion': np.concatenate([l['emotion'] for l in true_labels]),
            'sub_emotion': np.concatenate([l['sub_emotion'] for l in true_labels]),
            'intensity': np.concatenate([l['intensity'] for l in true_labels])
        }
        
        all_probs = {
            'emotion': np.concatenate([p['emotion'] for p in probabilities]),
            'sub_emotion': np.concatenate([p['sub_emotion'] for p in probabilities]),
            'intensity': np.concatenate([p['intensity'] for p in probabilities])
        }
        
        # Decode numeric labels to text labels using label encoders
        pred_emotion = self.label_encoders['emotion'].inverse_transform(all_preds['emotion'])
        true_emotion = self.label_encoders['emotion'].inverse_transform(all_labels['emotion'])
        pred_sub_emotion = self.label_encoders['sub_emotion'].inverse_transform(all_preds['sub_emotion'])
        true_sub_emotion = self.label_encoders['sub_emotion'].inverse_transform(all_labels['sub_emotion'])
        pred_intensity = self.label_encoders['intensity'].inverse_transform(all_preds['intensity'])
        true_intensity = self.label_encoders['intensity'].inverse_transform(all_labels['intensity'])
        
        # Create comprehensive results DataFrame
        self.results_df = pd.DataFrame({
            'text': texts,
            'true_emotion': true_emotion,
            'pred_emotion': pred_emotion,
            'true_sub_emotion': true_sub_emotion,
            'pred_sub_emotion': pred_sub_emotion,
            'true_intensity': true_intensity,
            'pred_intensity': pred_intensity,
            'emotion_confidence': np.max(all_probs['emotion'], axis=1),
            'sub_emotion_confidence': np.max(all_probs['sub_emotion'], axis=1),
            'intensity_confidence': np.max(all_probs['intensity'], axis=1),
            'text_length': [len(text) for text in texts]
        })
        
        # Add correctness columns for each task
        self.results_df['emotion_correct'] = self.results_df['true_emotion'] == self.results_df['pred_emotion']
        self.results_df['sub_emotion_correct'] = self.results_df['true_sub_emotion'] == self.results_df['pred_sub_emotion']
        self.results_df['intensity_correct'] = self.results_df['true_intensity'] == self.results_df['pred_intensity']
        self.results_df['all_correct'] = self.results_df['emotion_correct'] & self.results_df['sub_emotion_correct'] & self.results_df['intensity_correct']
        
        # Calculate confidence-based metrics
        self.results_df['confidence_level'] = pd.qcut(
            self.results_df[['emotion_confidence', 'sub_emotion_confidence', 'intensity_confidence']].mean(axis=1),
            q=5,
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
    
        # Separate correct and incorrect predictions
        self.correct_df = self.results_df[self.results_df['all_correct']]
        self.error_df = self.results_df[~self.results_df['all_correct']]
        
        # Calculate performance metrics
        self._calculate_performance_metrics()
        
        return self.results_df, self.correct_df, self.error_df

    def _calculate_performance_metrics(self):
        """
        Calculate comprehensive performance metrics for the model.
        
        This method computes various performance metrics including:
        1. Overall accuracy and error counts
        2. Task-specific accuracies and confidences
        3. Confidence-based performance analysis
        
        Note:
            This method should be called after analyze_errors() has been executed.
            It populates the self.performance_metrics attribute with detailed statistics.
        """
        if self.results_df is None:
            raise ValueError("Please run analyze_errors() first")
            
        self.performance_metrics = {
            'overall': {
                'total_examples': len(self.results_df),
                'correct_examples': len(self.correct_df),
                'error_examples': len(self.error_df),
                'accuracy': self.results_df['all_correct'].mean()
            },
            'task_specific': {
                'emotion': {
                    'accuracy': self.results_df['emotion_correct'].mean(),
                    'confidence': self.results_df['emotion_confidence'].mean()
                },
                'sub_emotion': {
                    'accuracy': self.results_df['sub_emotion_correct'].mean(),
                    'confidence': self.results_df['sub_emotion_confidence'].mean()
                },
                'intensity': {
                    'accuracy': self.results_df['intensity_correct'].mean(),
                    'confidence': self.results_df['intensity_confidence'].mean()
                }
            },
            'confidence_analysis': {
                level: {
                    'count': len(self.results_df[self.results_df['confidence_level'] == level]),
                    'accuracy': self.results_df[self.results_df['confidence_level'] == level]['all_correct'].mean()
                }
                for level in self.results_df['confidence_level'].unique()
            }
        }

    def analyze_error_patterns(self) -> Dict:
        """
        Analyze patterns in the errors made by the model.
        
        This method performs a detailed analysis of error patterns, including:
        1. Error type distribution (single vs. multiple task errors)
        2. Most common emotion misclassifications
        3. Most common sub-emotion misclassifications
        4. Most common intensity misclassifications
        5. Text length analysis
        6. Confidence vs. accuracy correlation
        
        Returns:
            Dict: A dictionary containing various error analyses:
                - error_counts: Distribution of different error types
                - emotion_confusion: Top 10 most common emotion misclassifications
                - sub_emotion_confusion: Top 10 most common sub-emotion misclassifications
                - intensity_confusion: Top 10 most common intensity misclassifications
                - length_stats: Performance statistics by text length
                - emotion_stats: Performance statistics by emotion category
                - confidence_correlation: Correlation between confidence and accuracy
        
        Note:
            This method should be called after analyze_errors() has been executed.
            It populates the self.error_analysis attribute with detailed statistics.
        """
        if self.error_df is None:
            raise ValueError("Please run analyze_errors() first")
            
        # Count number of errors by type (single vs. multiple task errors)
        error_counts = {
            'emotion_only': len(self.error_df[~self.error_df['emotion_correct'] & self.error_df['sub_emotion_correct'] & self.error_df['intensity_correct']]),
            'sub_emotion_only': len(self.error_df[self.error_df['emotion_correct'] & ~self.error_df['sub_emotion_correct'] & self.error_df['intensity_correct']]),
            'intensity_only': len(self.error_df[self.error_df['emotion_correct'] & self.error_df['sub_emotion_correct'] & ~self.error_df['intensity_correct']]),
            'emotion_and_sub': len(self.error_df[~self.error_df['emotion_correct'] & ~self.error_df['sub_emotion_correct'] & self.error_df['intensity_correct']]),
            'emotion_and_intensity': len(self.error_df[~self.error_df['emotion_correct'] & self.error_df['sub_emotion_correct'] & ~self.error_df['intensity_correct']]),
            'sub_and_intensity': len(self.error_df[self.error_df['emotion_correct'] & ~self.error_df['sub_emotion_correct'] & ~self.error_df['intensity_correct']]),
            'all_wrong': len(self.error_df[~self.error_df['emotion_correct'] & ~self.error_df['sub_emotion_correct'] & ~self.error_df['intensity_correct']])
        }
            
        # Analyze most common emotion misclassifications with confidence scores
        emotion_errors = self.error_df[~self.error_df['emotion_correct']]
        emotion_confusion = (
            emotion_errors.groupby(['true_emotion', 'pred_emotion'])
            .agg({
                'emotion_confidence': 'mean',
                'text': 'size'
            })
            .rename(columns={'emotion_confidence': 'avg_confidence', 'text': 'count'})
            .reset_index()
            .sort_values('count', ascending=False)
        )
        
        # Analyze most common sub-emotion misclassifications
        sub_emotion_errors = self.error_df[~self.error_df['sub_emotion_correct']]
        sub_emotion_confusion = (
            sub_emotion_errors.groupby(['true_sub_emotion', 'pred_sub_emotion'])
            .agg({
                'sub_emotion_confidence': 'mean',
                'text': 'size'
            })
            .rename(columns={'sub_emotion_confidence': 'avg_confidence', 'text': 'count'})
            .reset_index()
            .sort_values('count', ascending=False)
        )
        
        # Analyze most common intensity misclassifications
        intensity_errors = self.error_df[~self.error_df['intensity_correct']]
        intensity_confusion = (
            intensity_errors.groupby(['true_intensity', 'pred_intensity'])
            .agg({
                'intensity_confidence': 'mean',
                'text': 'size'
            })
            .rename(columns={'intensity_confidence': 'avg_confidence', 'text': 'count'})
            .reset_index()
            .sort_values('count', ascending=False)
        )

        # Analyze performance by text length
        bins = [0, 50, 100, 150, 200, 300, 400, 500, 1000, float('inf')]
        bin_labels = ['0-50', '51-100', '101-150', '151-200', '201-300', 
                    '301-400', '401-500', '501-1000', '1000+']

        # Use results_df for length analysis to include all examples
        self.results_df['length_bin'] = pd.cut(self.results_df['text_length'], bins=bins, labels=bin_labels)
        
        # Calculate comprehensive length statistics
        length_stats = self.results_df.groupby('length_bin').agg({
            'all_correct': ['mean', 'count'],
            'emotion_confidence': 'mean',
            'sub_emotion_confidence': 'mean',
            'intensity_confidence': 'mean'
        }).round(3)
        
        # Calculate per-emotion accuracy and confidence
        emotion_stats = self.results_df.groupby('true_emotion').agg({
            'emotion_correct': 'mean',
            'emotion_confidence': 'mean',
            'text': 'count'
        }).rename(columns={'text': 'count'}).round(3)
        
        # Calculate confidence vs accuracy correlation for each task
        confidence_correlation = {
            'emotion': self.results_df['emotion_confidence'].corr(self.results_df['emotion_correct'].astype(int)),
            'sub_emotion': self.results_df['sub_emotion_confidence'].corr(self.results_df['sub_emotion_correct'].astype(int)),
            'intensity': self.results_df['intensity_confidence'].corr(self.results_df['intensity_correct'].astype(int))
        }
        
        # Store all analysis results
        self.error_analysis = {
            'error_counts': error_counts,
            'emotion_confusion': emotion_confusion.head(10),
            'sub_emotion_confusion': sub_emotion_confusion.head(10),
            'intensity_confusion': intensity_confusion.head(10),
            'length_stats': length_stats,
            'emotion_stats': emotion_stats,
            'confidence_correlation': confidence_correlation
        }
        
        return self.error_analysis

    def visualize_error_patterns(self):
        """
        Visualize common error patterns with enhanced plots.
        
        This method creates a comprehensive visualization dashboard including:
        1. Distribution of error types
        2. Top emotion misclassifications
        3. Confidence vs accuracy correlation
        4. Accuracy by text length
        5. Accuracy by emotion category
        6. Example misclassifications table
        
        The visualization uses a modern color scheme and professional styling,
        with enhanced readability and informative annotations.
        
        Note:
            This method should be called after analyze_error_patterns() has been executed.
            It saves the visualization to a file if save_plots is enabled in the config.
        """
        if self.error_analysis is None:
            raise ValueError("Please run analyze_error_patterns() first")
            
        # Set style for better-looking plots
        plt.style.use('ggplot')
        
        # Create a figure with subplots using a modern color scheme
        fig = plt.figure(figsize=(22, 20))
        
        # Configure grid layout with appropriate spacing
        gs = fig.add_gridspec(3, 2, 
                            hspace=0.6,  # Vertical spacing between plots
                            wspace=0.4,  # Horizontal spacing between plots
                            height_ratios=[1, 1, 1.2],  # Space allocation for rows
                            top=0.95,    # Top margin
                            bottom=0.05, # Bottom margin
                            left=0.1,    # Left margin
                            right=0.9)   # Right margin
        
        # Define color palettes for different plots
        error_palette = sns.color_palette("RdYlBu_r", n_colors=7)
        emotion_palette = sns.color_palette("viridis", n_colors=10)
        corr_palette = sns.color_palette("coolwarm", n_colors=3)
        accuracy_palette = sns.color_palette("mako", n_colors=9)
        
        # Add overall title
        fig.suptitle('Emotion Classification Error Analysis', 
                    fontsize=self.config.font_size['title']+4,
                    y=0.98,
                    weight='bold')
        
        # Create and style each subplot
        # 1. Error Types Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        counts = self.error_analysis['error_counts']
        bars = sns.barplot(x=list(counts.keys()), y=list(counts.values()), ax=ax1, palette=error_palette)
        ax1.set_title('Distribution of Error Types', fontsize=self.config.font_size['title'], pad=20)
        ax1.tick_params(axis='x', rotation=45, labelsize=10)
        plt.setp(ax1.get_xticklabels(), ha='right')
        ax1.set_ylabel('Count', fontsize=self.config.font_size['label'])
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add count labels
        for i, v in enumerate(counts.values()):
            if np.isfinite(v):
                ax1.text(i, v + 1, str(v), ha='center', va='bottom',
                        fontsize=self.config.font_size['annotation'],
                        fontweight='bold')
        
        # 2. Top Emotion Misclassifications
        ax2 = fig.add_subplot(gs[0, 1])
        error_pairs = self.error_analysis['emotion_confusion'].head(10)
        pair_labels = [f"{t}→{p}" for t, p in zip(error_pairs['true_emotion'], error_pairs['pred_emotion'])]
        bars = sns.barplot(x=pair_labels, y=error_pairs['count'], ax=ax2, palette=emotion_palette)
        ax2.set_title('Top 10 Emotion Misclassifications', fontsize=self.config.font_size['title'], pad=20)
        ax2.tick_params(axis='x', rotation=45, labelsize=10)
        plt.setp(ax2.get_xticklabels(), ha='right')
        ax2.set_ylabel('Count', fontsize=self.config.font_size['label'])
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add count labels
        for i, v in enumerate(error_pairs['count']):
            if np.isfinite(v):
                ax2.text(i, v + 0.5, str(int(v)), ha='center', va='bottom',
                        fontsize=self.config.font_size['annotation'],
                        fontweight='bold')
        
        # 3. Confidence vs Accuracy Correlation
        ax3 = fig.add_subplot(gs[1, 0])
        confidence_corr = self.error_analysis['confidence_correlation']
        bars = sns.barplot(x=list(confidence_corr.keys()), y=list(confidence_corr.values()),
                          ax=ax3, palette=corr_palette)
        ax3.set_title('Confidence vs Accuracy Correlation', fontsize=self.config.font_size['title'], pad=20)
        ax3.set_ylabel('Correlation Coefficient', fontsize=self.config.font_size['label'])
        ax3.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add correlation labels with color coding
        for i, v in enumerate(confidence_corr.values()):
            if np.isfinite(v):
                color = 'green' if v > 0 else 'red'
                ax3.text(i, v + 0.02, f"{v:.3f}", ha='center', va='bottom',
                        fontsize=self.config.font_size['annotation'],
                        fontweight='bold', color=color)
        
        # 4. Accuracy by Text Length
        ax4 = fig.add_subplot(gs[1, 1])
        length_stats = self.error_analysis['length_stats']
        length_stats_mean = length_stats[('all_correct', 'mean')].fillna(0)
        bars = sns.barplot(x=length_stats.index, y=length_stats_mean, ax=ax4,
                          palette=sns.color_palette("viridis", n_colors=len(length_stats_mean)))
        ax4.set_title('Accuracy by Text Length', fontsize=self.config.font_size['title'], pad=20)
        ax4.tick_params(axis='x', rotation=45, labelsize=10)
        plt.setp(ax4.get_xticklabels(), ha='right')
        ax4.set_ylabel('Accuracy', fontsize=self.config.font_size['label'])
        ax4.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add accuracy labels
        for i, v in enumerate(length_stats_mean):
            if np.isfinite(v):
                ax4.text(i, v + 0.02, f"{v:.3f}", ha='center', va='bottom',
                        fontsize=self.config.font_size['annotation'],
                        fontweight='bold')
        
        # 5. Accuracy by Emotion Category
        ax5 = fig.add_subplot(gs[2, 0])
        emotion_stats = self.error_analysis['emotion_stats']
        emotion_correct = emotion_stats['emotion_correct'].fillna(0)
        bars = sns.barplot(x=emotion_stats.index, y=emotion_correct, ax=ax5,
                          palette=accuracy_palette)
        ax5.set_title('Accuracy by Emotion Category', fontsize=self.config.font_size['title'], pad=20)
        ax5.tick_params(axis='x', rotation=45, labelsize=10)
        plt.setp(ax5.get_xticklabels(), ha='right')
        ax5.set_ylabel('Accuracy', fontsize=self.config.font_size['label'])
        ax5.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add accuracy labels
        for i, v in enumerate(emotion_correct):
            if np.isfinite(v):
                ax5.text(i, v + 0.02, f"{v:.3f}", ha='center', va='bottom',
                        fontsize=self.config.font_size['annotation'],
                        fontweight='bold')
        
        # 6. Example Misclassifications Table
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')
        example_count = min(self.config.example_count, len(self.error_df))
        error_examples = self.error_df.sample(example_count)
        
        # Prepare table data
        table_data = []
        for _, row in error_examples.iterrows():
            text = row['text']
            if len(text) > self.config.max_text_length:
                text = text[:self.config.max_text_length-3] + "..."
            
            confidence = row['emotion_confidence']
            confidence_str = f"{confidence:.2f}" if np.isfinite(confidence) else "N/A"
            
            table_data.append([
                text,
                row['true_emotion'],
                row['pred_emotion'],
                confidence_str
            ])
        
        # Create and style the table
        table = ax6.table(
            cellText=table_data,
            colLabels=['Text', 'True', 'Pred', 'Conf'],
            loc='center',
            cellLoc='left',
            colColours=['#f0f0f0']*4
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)
        
        # Add cell colors and styling
        for key, cell in table._cells.items():
            if key[0] == 0:  # Header row
                cell.set_text_props(weight='bold', color='black')
                cell.set_facecolor('#e0e0e0')
            else:  # Data rows
                cell.set_facecolor('#f9f9f9' if key[0] % 2 == 0 else 'white')
                cell.set_text_props(color='black')
            cell.set_edgecolor('#cccccc')
        
        # Add table title
        ax6.set_title('Example Misclassifications', fontsize=self.config.font_size['title'], pad=20)
        
        # Save the plot if configured
        if self.config.save_plots:
            plt.savefig(
                os.path.join(self.config.output_dir, f'error_analysis_{self.timestamp}.{self.config.plot_format}'),
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                pad_inches=0.5
            )
        plt.show()

    def examine_specific_errors(self, emotion_pair: Optional[Tuple[str, str]] = None, n_examples: int = 5) -> pd.DataFrame:
        """
        Examine specific error cases for detailed analysis.
        
        This method allows for detailed examination of specific error cases, either:
        1. For a specific emotion confusion pair (true emotion → predicted emotion)
        2. For the most common error pattern in the dataset
        
        Args:
            emotion_pair (Optional[Tuple[str, str]]): Tuple of (true_emotion, pred_emotion) to filter for.
                                                    If None, examines the most common error pattern.
            n_examples (int): Number of example errors to display.
        
        Returns:
            pd.DataFrame: DataFrame containing filtered errors with detailed information.
        
        Note:
            For each example, displays:
            - Truncated text content
            - Text length
            - Prediction confidence
            - True and predicted labels for all tasks
            - Sub-emotion and intensity predictions
        """
        if self.error_df is None:
            raise ValueError("Please run analyze_errors() first")
            
        # Filter errors based on specified emotion pair or most common error
        if emotion_pair:
            true_emotion, pred_emotion = emotion_pair
            filtered_errors = self.error_df[
                (self.error_df['true_emotion'] == true_emotion) & 
                (self.error_df['pred_emotion'] == pred_emotion)
            ]
            print(f"\n{'='*80}")
            print(f"Examples where true emotion '{true_emotion}' was predicted as '{pred_emotion}'")
            print(f"Total instances: {len(filtered_errors)}")
            print(f"Average confidence: {filtered_errors['emotion_confidence'].mean():.3f}")
            print(f"{'='*80}")
        else:
            # Get the most common error pattern
            common_error = self.error_df[~self.error_df['emotion_correct']].groupby(['true_emotion', 'pred_emotion']).size().reset_index(name='count')
            common_error = common_error.sort_values('count', ascending=False).iloc[0]
            true_emotion, pred_emotion = common_error['true_emotion'], common_error['pred_emotion']
            filtered_errors = self.error_df[
                (self.error_df['true_emotion'] == true_emotion) & 
                (self.error_df['pred_emotion'] == pred_emotion)
            ]
            print(f"\n{'='*80}")
            print(f"Most common error: true emotion '{true_emotion}' predicted as '{pred_emotion}'")
            print(f"Total instances: {len(filtered_errors)}")
            print(f"Average confidence: {filtered_errors['emotion_confidence'].mean():.3f}")
            print(f"{'='*80}")
            
        # Display example errors with detailed information
        examples = filtered_errors.sample(min(n_examples, len(filtered_errors)))
        for i, (_, row) in enumerate(examples.iterrows()):
            print(f"\nExample {i+1}:")
            print(f"{'-'*40}")
            print(f"Text: {row['text'][:100]}...")
            print(f"Length: {row['text_length']} characters")
            print(f"Confidence: {row['emotion_confidence']:.3f}")
            print(f"True emotion: {row['true_emotion']}")
            print(f"Predicted: {row['pred_emotion']}")
            print(f"Sub-emotion: {row['true_sub_emotion']} → {row['pred_sub_emotion']}")
            print(f"Intensity: {row['true_intensity']} → {row['pred_intensity']}")
            print(f"{'-'*40}")
        
        return filtered_errors

    def generate_recommendations(self) -> Dict:
        """
        Generate comprehensive recommendations for model improvement.
        
        This method analyzes error patterns and performance metrics to generate
        actionable recommendations for improving the model. It covers:
        1. Most problematic emotions
        2. Commonly confused emotion pairs
        3. Text length analysis
        4. Confidence calibration
        5. Hierarchical classification improvements
        
        Returns:
            Dict: A dictionary containing:
                - summary: Overall statistics and timestamp
                - recommendations: List of detailed recommendations with supporting data
        
        Note:
            Each recommendation includes:
            - Type of improvement
            - Supporting statistics
            - Specific suggestions for implementation
        """
        if self.error_df is None or self.error_analysis is None:
            raise ValueError("Please run analyze_errors() and analyze_error_patterns() first")
            
        recommendations = {
            'summary': {
                'total_errors': len(self.error_df),
                'overall_accuracy': self.performance_metrics['overall']['accuracy'],
                'timestamp': self.timestamp
            },
            'recommendations': []
        }
        
        # 1. Analyze most problematic emotions
        prob_emotions = self.error_df[~self.error_df['emotion_correct']]['true_emotion'].value_counts()
        most_problematic = prob_emotions.index[0]
        recommendations['recommendations'].append({
            'type': 'problematic_emotion',
            'emotion': most_problematic,
            'error_count': prob_emotions[0],
            'error_rate': prob_emotions[0] / len(self.error_df),
            'suggestions': [
                'Add more training examples',
                'Use data augmentation',
                'Consider class weights in loss function'
            ]
        })
        
        # 2. Analyze commonly confused emotion pairs
        emotion_confusion = self.error_analysis['emotion_confusion']
        top_confusion = emotion_confusion.iloc[0]
        recommendations['recommendations'].append({
            'type': 'emotion_confusion',
            'true_emotion': top_confusion['true_emotion'],
            'predicted_emotion': top_confusion['pred_emotion'],
            'count': top_confusion['count'],
            'avg_confidence': top_confusion['avg_confidence'],
            'suggestions': [
                'Feature engineering to distinguish emotions',
                'Add contrastive learning',
                'Review training data quality'
            ]
        })
        
        # 3. Analyze text length impact
        length_corr = self.error_df['text_length'].corr(self.error_df['all_correct'].astype(int))
        recommendations['recommendations'].append({
            'type': 'text_length',
            'correlation': length_corr,
            'suggestions': [
                'Increase max sequence length' if length_corr < 0 else 'Data augmentation for short examples',
                'Consider hierarchical attention',
                'Review tokenization strategy'
            ]
        })
        
        # 4. Analyze confidence calibration
        confidence_corr = self.error_analysis['confidence_correlation']
        recommendations['recommendations'].append({
            'type': 'confidence',
            'correlations': confidence_corr,
            'suggestions': [
                'Calibrate model confidence',
                'Implement confidence thresholding',
                'Review loss function'
            ]
        })
        
        # 5. Analyze hierarchical classification
        sub_emotion_acc = self.performance_metrics['task_specific']['sub_emotion']['accuracy']
        intensity_acc = self.performance_metrics['task_specific']['intensity']['accuracy']
        
        recommendations['recommendations'].append({
            'type': 'hierarchical',
            'sub_emotion_accuracy': sub_emotion_acc,
            'intensity_accuracy': intensity_acc,
            'suggestions': [
                'Implement hierarchical classification',
                'Add task-specific attention layers',
                'Review label hierarchy'
            ]
        })
        
        # Print recommendations in a formatted way
        print(f"\n{'='*80}")
        print(f"{'MODEL IMPROVEMENT RECOMMENDATIONS':^80}")
        print(f"{'='*80}")
        print(f"Analysis timestamp: {self.timestamp}")
        print(f"Total errors: {len(self.error_df)}")
        print(f"Overall accuracy: {self.performance_metrics['overall']['accuracy']:.3f}")
        print(f"\n{'='*80}")
        
        for rec in recommendations['recommendations']:
            print(f"\n{rec['type'].upper()} ANALYSIS:")
            print(f"{'-'*40}")
            if rec['type'] == 'problematic_emotion':
                print(f"Focus on '{rec['emotion']}' emotion:")
                print(f"→ {rec['error_count']} errors ({rec['error_rate']:.2%})")
            elif rec['type'] == 'emotion_confusion':
                print(f"Address confusion between '{rec['true_emotion']}' and '{rec['predicted_emotion']}':")
                print(f"→ {rec['count']} instances (avg confidence: {rec['avg_confidence']:.3f})")
            elif rec['type'] == 'text_length':
                print(f"Text length correlation: {rec['correlation']:.3f}")
            elif rec['type'] == 'confidence':
                print("Confidence correlations:")
                for task, corr in rec['correlations'].items():
                    print(f"→ {task}: {corr:.3f}")
            elif rec['type'] == 'hierarchical':
                print(f"Sub-emotion accuracy: {rec['sub_emotion_accuracy']:.3f}")
                print(f"Intensity accuracy: {rec['intensity_accuracy']:.3f}")
            
            print("\nSuggested improvements:")
            for suggestion in rec['suggestions']:
                print(f"→ {suggestion}")
        
        return recommendations

    def save_results(self, iteration_num: int):
        """
        Save comprehensive analysis results to files.
        
        This method saves all analysis results, including:
        1. Misclassified examples
        2. Summary statistics
        3. Performance metrics
        4. Error analysis
        5. Recommendations
        
        The results are saved in both CSV and JSON formats for different use cases.
        
        Args:
            iteration_num (int): Current iteration number for file naming
        
        Note:
            Creates an iteration-specific directory and saves:
            - misclassified_examples.csv: Detailed error cases
            - summary_stats.csv: Overall statistics
            - analysis_results.json: Complete analysis results
            - recommendations.json: Improvement recommendations
        """
        if self.error_df is None:
            raise ValueError("Please run analyze_errors() first")
            
        # Create iteration-specific directory
        iteration_dir = os.path.join(self.config.output_dir, f'iteration_{iteration_num}')
        os.makedirs(iteration_dir, exist_ok=True)
        
        # Save misclassified examples
        self.error_df.to_csv(
            os.path.join(iteration_dir, 'misclassified_examples.csv'),
            index=False
        )
        
        def convert_to_serializable(obj):
            """
            Convert numpy types to Python native types for JSON serialization.
            
            Args:
                obj: Object to convert
            
            Returns:
                Converted object that can be serialized to JSON
            """
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                np.int16, np.int32, np.int64, np.uint8,
                np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {str(k) if isinstance(k, tuple) else k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return str(obj)
            return obj

        # Prepare summary statistics
        summary_stats = {
            'timestamp': self.timestamp,
            'iteration': iteration_num,
            'total_examples': int(len(self.results_df)),
            'correct_examples': int(len(self.correct_df)),
            'error_examples': int(len(self.error_df)),
            'emotion_accuracy': float(self.results_df['emotion_correct'].mean()),
            'sub_emotion_accuracy': float(self.results_df['sub_emotion_correct'].mean()),
            'intensity_accuracy': float(self.results_df['intensity_correct'].mean()),
            'overall_accuracy': float(self.results_df['all_correct'].mean())
        }

        # Convert performance metrics to serializable format
        serializable_metrics = {}
        for category, metrics in self.performance_metrics.items():
            if isinstance(metrics, dict):
                serializable_metrics[str(category)] = {
                    str(k): convert_to_serializable(v) for k, v in metrics.items()
                }
            else:
                serializable_metrics[str(category)] = convert_to_serializable(metrics)
        
        summary_stats['performance_metrics'] = serializable_metrics

        # Convert error analysis to serializable format
        serializable_analysis = {}
        for key, value in self.error_analysis.items():
            if isinstance(value, pd.DataFrame):
                serializable_analysis[str(key)] = value.to_dict('records')
            else:
                serializable_analysis[str(key)] = convert_to_serializable(value)
        
        summary_stats['error_analysis'] = serializable_analysis
        
        # Save as both CSV and JSON
        pd.DataFrame([summary_stats]).to_csv(
            os.path.join(iteration_dir, 'summary_stats.csv'),
            index=False
        )
        
        # Ensure all dictionary keys are strings for JSON serialization
        def ensure_string_keys(d):
            if isinstance(d, dict):
                return {str(k): ensure_string_keys(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [ensure_string_keys(item) for item in d]
            return d

        with open(os.path.join(iteration_dir, 'analysis_results.json'), 'w') as f:
            json.dump(ensure_string_keys(summary_stats), f, indent=4)
        
        # Generate and save recommendations
        recommendations = self.generate_recommendations()
        
        # Convert recommendations to serializable format
        serializable_recommendations = {
            'summary': {k: convert_to_serializable(v) for k, v in recommendations['summary'].items()},
            'recommendations': []
        }
        
        for rec in recommendations['recommendations']:
            serializable_rec = {}
            for k, v in rec.items():
                serializable_rec[k] = convert_to_serializable(v)
            serializable_recommendations['recommendations'].append(serializable_rec)
        
        with open(os.path.join(iteration_dir, 'recommendations.json'), 'w') as f:
            json.dump(serializable_recommendations, f, indent=4)
        
        print(f"\nResults saved to: {iteration_dir}")

# Example usage:
"""
# Initialize the error analysis with custom configuration
config = AnalysisConfig(
    figure_size=(20, 15),
    font_size={'title': 16, 'label': 12, 'tick': 10, 'annotation': 10},
    color_palette="viridis",
    max_text_length=70,
    example_count=30,
    correlation_threshold=0.1,
    save_plots=True,
    plot_format="png",
    output_dir="./results/error_analysis"
)

error_analyzer = ErrorAnalysis(
    model=model,
    test_dataloader=test_dataloader,
    device=device,
    train_df=train_df,
    test_df=test_df,
    config=config
)

# Run the analysis
results_df, correct_df, error_df = error_analyzer.analyze_errors()
error_analysis = error_analyzer.analyze_error_patterns()

# Visualize results
error_analyzer.visualize_error_patterns()

# Examine specific errors
error_analyzer.examine_specific_errors()

# Generate recommendations
recommendations = error_analyzer.generate_recommendations()

# Save results
error_analyzer.save_results(iteration_num)
"""



