"""
Custom trainer class for BERT-based emotion classification model.
This module provides a comprehensive training and evaluation framework for multi-task emotion classification,
including emotion, sub-emotion, and intensity prediction with flexible output options.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import functional as F
from transformers import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tabulate import tabulate
from termcolor import colored
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tqdm import tqdm
import pickle
import os



class CustomTrainer:
    """
    A custom trainer class for BERT-based emotion classification model with flexible outputs.
    
    This class handles the complete training pipeline including:
    - Model training with customizable multi-task learning
    - Validation and testing with flexible output options
    - Performance metrics calculation and visualization
    - Feature importance analysis
    
    Attributes:
        model (nn.Module): The BERT-based emotion classification model
        train_dataloader (DataLoader): DataLoader for training data
        val_dataloader (DataLoader): DataLoader for validation data
        test_dataloader (DataLoader): DataLoader for test data
        device (torch.device): Device to run the model on (CPU/GPU)
        test_set (Dataset): Test dataset
        feature_dim (int): Dimension of input features (automatically determined from data)
        class_weights_tensor (torch.Tensor): Class weights for handling class imbalance
        iteration_num (int): Current iteration number for model saving
        output_tasks (list): List of tasks to output (e.g., ['emotion', 'sub_emotion', 'intensity'])
    """
    
    def __init__(self, model, train_dataloader, val_dataloader, test_dataloader, 
                 device, test_set, class_weights_tensor,
                 iteration_num, encoders_dir, output_tasks=None):
        """
        Initialize the CustomTrainer with model and data components.
        
        Args:
            model (nn.Module): The BERT-based emotion classification model
            train_dataloader (DataLoader): DataLoader for training data
            val_dataloader (DataLoader): DataLoader for validation data
            test_dataloader (DataLoader): DataLoader for test data
            device (torch.device): Device to run the model on (CPU/GPU)
            test_set (Dataset): Test dataset
            class_weights_tensor (torch.Tensor): Class weights for handling class imbalance
            iteration_num (int): Current iteration number for model saving
            encoders_dir (str): Directory containing the encoder pickle files
            output_tasks (list, optional): List of tasks to output. Defaults to ['emotion', 'sub_emotion', 'intensity']
        """
        # Store model and data components
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.test_set = test_set
        self.class_weights_tensor = class_weights_tensor
        self.iteration_num = iteration_num
        
        # Set output tasks
        self.output_tasks = output_tasks or ['emotion', 'sub_emotion', 'intensity']
        
        # Load encoders
        self._load_encoders(encoders_dir)
        
        # Automatically determine feature dimension from the first batch
        self.feature_dim = self._get_feature_dim()
        
        # Training hyperparameters
        self.learning_rate = 2e-5  # Learning rate for AdamW optimizer
        self.weight_decay = 0.01   # Weight decay for regularization
        self.epochs = 5            # Number of training epochs
        
        # Task weights for multi-task learning
        self.task_weights = {
            'emotion': 1.0 if 'emotion' in self.output_tasks else 0.0,
            'sub_emotion': 0.8 if 'sub_emotion' in self.output_tasks else 0.0,
            'intensity': 0.2 if 'intensity' in self.output_tasks else 0.0
        }

    def _get_feature_dim(self):
        """
        Determine the feature dimension from the first batch of training data.
        
        Returns:
            int: Dimension of the feature vector
        """
        # Get the first batch
        first_batch = next(iter(self.train_dataloader))
        
        # Get the feature dimension from the features tensor
        feature_dim = first_batch['features'].shape[-1]
        
        return feature_dim

    def _load_encoders(self, encoders_dir):
        """
        Load label encoders from pickle files.
        
        Args:
            encoders_dir (str): Directory containing the encoder pickle files
        """
        # Load emotion encoder
        with open(f'{encoders_dir}/emotion_encoder.pkl', 'rb') as f:
            self.emotion_encoder = pickle.load(f)
            
        # Load sub-emotion encoder
        with open(f'{encoders_dir}/sub_emotion_encoder.pkl', 'rb') as f:
            self.sub_emotion_encoder = pickle.load(f)
            
        # Load intensity encoder
        with open(f'{encoders_dir}/intensity_encoder.pkl', 'rb') as f:
            self.intensity_encoder = pickle.load(f)

    def setup_training(self):
        """
        Set up training components including loss function, optimizer, and learning rate scheduler.
        
        Returns:
            tuple: (criterion_dict, optimizer, scheduler)
                - criterion_dict: Dictionary of loss functions for each task
                - optimizer: AdamW optimizer with weight decay
                - scheduler: Linear learning rate scheduler with warmup
        """
        # Initialize loss functions with appropriate class weights for each task
        criterion_dict = {}
        
        # For emotion task - use the provided class weights
        if 'emotion' in self.output_tasks:
            criterion_dict['emotion'] = nn.CrossEntropyLoss(weight=self.class_weights_tensor)
        
        # For sub-emotion and intensity - use regular CrossEntropyLoss without weights
        if 'sub_emotion' in self.output_tasks:
            criterion_dict['sub_emotion'] = nn.CrossEntropyLoss()
        
        if 'intensity' in self.output_tasks:
            criterion_dict['intensity'] = nn.CrossEntropyLoss()
        
        # Initialize AdamW optimizer with weight decay for regularization
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        # Calculate total training steps for scheduler
        total_steps = len(self.train_dataloader) * self.epochs
        
        # Initialize learning rate scheduler with warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0.1 * total_steps,  # 10% warmup
            num_training_steps=total_steps
        )
        
        return criterion_dict, optimizer, scheduler

    def train_epoch(self, criterion_dict, optimizer, scheduler):
        """
        Train the model for one epoch.
        
        Args:
            criterion_dict (dict): Dictionary of loss functions for each task
            optimizer (torch.optim.Optimizer): Optimizer
            scheduler (torch.optim.lr_scheduler): Learning rate scheduler
            
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()  # Set model to training mode
        train_loss = 0
        
        # Iterate over training batches
        for batch in tqdm(self.train_dataloader, desc="Training", ncols=120, colour='green'):
            # Move batch data to device (CPU/GPU)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            features = batch['features'].to(self.device)
            
            # Get labels for selected tasks
            labels = {}
            for task in self.output_tasks:
                labels[task] = batch[f'{task}_label'].to(self.device)
            
            # Forward pass through the model
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                features=features
            )

            # Handle single output vs multiple outputs
            if len(self.output_tasks) == 1 and not isinstance(outputs, (list, tuple)):
                outputs = [outputs]
            
            # Calculate losses for selected tasks
            loss = 0
            for i, task in enumerate(self.output_tasks):
                # Ensure output has proper batch dimension
                output = outputs[i]
                if output.dim() == 1:
                    output = output.unsqueeze(0)
                    
                task_loss = criterion_dict[task](output, labels[task])
                loss += self.task_weights[task] * task_loss
            
            # Backward pass and optimization
            optimizer.zero_grad()  # Clear gradients
            loss.backward()        # Compute gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()       # Update parameters
            scheduler.step()       # Update learning rate
            
            train_loss += loss.item()
        
        return train_loss / len(self.train_dataloader)  # Return average loss

    def evaluate(self, dataloader, criterion_dict, is_test=False):
        """
        Evaluate the model on validation or test data.
        
        Args:
            dataloader (DataLoader): DataLoader for evaluation data
            criterion_dict (dict): Dictionary of loss functions for each task
            is_test (bool): Whether this is test set evaluation
            
        Returns:
            tuple: (average_loss, predictions_dict, labels_dict)
                - average_loss: Average evaluation loss
                - predictions_dict: Dictionary containing predictions for selected tasks
                - labels_dict: Dictionary containing true labels for selected tasks
        """
        self.model.eval()  # Set model to evaluation mode
        val_loss = 0
        all_preds = {task: [] for task in self.output_tasks}
        all_labels = {task: [] for task in self.output_tasks}
        
        with torch.no_grad():  # Disable gradient computation
            for batch in tqdm(dataloader, desc="Testing" if is_test else "Validation", 
                            ncols=120, colour='orange' if is_test else 'blue'):
                # Move batch data to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                features = batch['features'].to(self.device)
                
                # Get labels for selected tasks
                labels = {}
                for task in self.output_tasks:
                    labels[task] = batch[f'{task}_label'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    features=features
                )

                # Handle single output vs multiple outputs
                if len(self.output_tasks) == 1 and not isinstance(outputs, (list, tuple)):
                    outputs = [outputs]
                
                # Calculate losses for selected tasks
                loss = 0
                for i, task in enumerate(self.output_tasks):
                    # Ensure output has proper batch dimension
                    output = outputs[i]
                    if output.dim() == 1:
                        output = output.unsqueeze(0)
                    
                    task_loss = criterion_dict[task](output, labels[task])
                    loss += self.task_weights[task] * task_loss
                    
                    # Collect predictions and labels
                    all_preds[task].extend(torch.argmax(outputs[i], dim=1).cpu().numpy())
                    all_labels[task].extend(labels[task].cpu().numpy())
                
                val_loss += loss.item()
        
        return val_loss / len(dataloader), all_preds, all_labels

    def train_and_evaluate(self):
        """
        Main training and evaluation loop.
        
        This method:
        1. Sets up training components
        2. Trains the model for multiple epochs
        3. Evaluates on validation and test sets
        4. Saves best models based on F1 scores for each task
        5. Prints metrics for each epoch
        """
        criterion_dict, optimizer, scheduler = self.setup_training()
        
        # Initialize best model tracking for each task
        best_val_f1s = {task: 0.0 for task in self.output_tasks}
        best_test_f1s = {task: 0.0 for task in self.output_tasks}
        
        # Training loop
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            
            # Training phase
            train_loss = self.train_epoch(criterion_dict, optimizer, scheduler)
            
            # Validation phase
            val_loss, val_preds, val_labels = self.evaluate(self.val_dataloader, criterion_dict)
            
            # Test phase
            _, test_preds, test_labels = self.evaluate(self.test_dataloader, criterion_dict, is_test=True)

            # Calculate metrics for selected tasks
            val_metrics = {}
            test_metrics = {}
            for task in self.output_tasks:
                val_metrics[task] = self.calculate_metrics(val_preds[task], val_labels[task])
                test_metrics[task] = self.calculate_metrics(test_preds[task], test_labels[task])
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}")
            self.print_metrics(val_metrics, "Val", val_loss)
            self.print_metrics(test_metrics, "Test")
            
            # Save best models for each task based on F1 scores
            for task in self.output_tasks:
                # Save based on validation F1
                if val_metrics[task]['f1'] > best_val_f1s[task]:
                    best_val_f1s[task] = val_metrics[task]['f1']
                    torch.save(self.model.state_dict(), 
                             f'./results/weights/best_val_in_{task}_f1_{val_metrics[task]["f1"]:.4f}_iteration_{self.iteration_num}.pt')
                    print(f"Model saved based on best validation F1 for {task}!")
                
                # Save based on test F1
                if test_metrics[task]['f1'] > best_test_f1s[task]:
                    best_test_f1s[task] = test_metrics[task]['f1']
                    torch.save(self.model.state_dict(), 
                             f'./results/weights/best_test_in_{task}_f1_{test_metrics[task]["f1"]:.4f}_iteration_{self.iteration_num}.pt')
                    print(f"Model saved based on best test F1 for {task}!")

    def evaluate_final_model(self):
        """
        Evaluate the final model and generate comprehensive visualizations.
        
        This method:
        1. Finds and loads the best model based on test F1 score for the current iteration
        2. Makes predictions on the test set
        3. Converts predictions to original labels
        4. Creates a results DataFrame
        5. Generates visualizations and analysis
        
        Returns:
            pd.DataFrame: DataFrame containing predictions and true labels
        """
        # Find the model with highest test F1 score for current iteration
        weights_dir = f'./results/weights'
        best_f1 = 0.0
        best_model_path = None
        
        # Loop through all model files for current iteration
        for filename in os.listdir(weights_dir):
            if f'iteration_{self.iteration_num}.pt' in filename and 'test' in filename:
                # Extract F1 score from filename
                f1_score = float(filename.split('f1_')[1].split('_')[0])
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_model_path = os.path.join(weights_dir, filename)
        
        if best_model_path is None:
            raise FileNotFoundError(f"No model files found for iteration {self.iteration_num}")
            
        print(f"Loading best model with test F1 score: {best_f1:.4f}")
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()

        # Initialize lists for predictions and labels
        predictions = {task: [] for task in self.output_tasks}
        labels = {task: [] for task in self.output_tasks}

        # Generate predictions
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, desc="Testing", ncols=120, colour="green"):
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

                # Handle single output vs multiple outputs
                if len(self.output_tasks) == 1 and not isinstance(outputs, (list, tuple)):
                    outputs = [outputs]
                
                # Get predictions and labels for selected tasks
                for i, task in enumerate(self.output_tasks):
                    pred = torch.argmax(outputs[i], dim=1).cpu().numpy()
                    label = batch[f'{task}_label'].cpu().numpy()
                    
                    predictions[task].extend(pred)
                    labels[task].extend(label)

        # Convert predictions and labels to original format
        results = {'text': self.test_set['text']}
        for task in self.output_tasks:
            encoder = getattr(self, f'{task}_encoder')
            results[f'true_{task}'] = encoder.inverse_transform(labels[task])
            results[f'pred_{task}'] = encoder.inverse_transform(predictions[task])

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        # Add correctness columns
        for task in self.output_tasks:
            results_df[f'{task}_correct'] = results_df[f'true_{task}'] == results_df[f'pred_{task}']
        
        if len(self.output_tasks) > 1:
            results_df['all_correct'] = True
            for task in self.output_tasks:
                results_df['all_correct'] &= results_df[f'{task}_correct']

        # Generate visualizations
        self._generate_visualizations(results_df)
        
        return results_df

    @staticmethod
    def calculate_metrics(preds, labels):
        """
        Calculate performance metrics for predictions.
        
        Args:
            preds (np.ndarray): Model predictions
            labels (np.ndarray): True labels
            
        Returns:
            dict: Dictionary containing accuracy, F1 score, precision, and recall
        """
        return {
            'acc': accuracy_score(labels, preds),
            'f1': f1_score(labels, preds, average='weighted'),
            'prec': precision_score(labels, preds, average='weighted'),
            'rec': recall_score(labels, preds, average='weighted')
        }

    @staticmethod
    def print_metrics(metrics_dict, split, loss=None):
        """
        Print formatted metrics with visual bars for better readability.
        
        Args:
            metrics_dict (dict): Dictionary containing metrics for each task
            split (str): Data split name (Train/Val/Test)
            loss (float, optional): Loss value to display
        """
        # Define colors for different splits
        split_colors = {
            'Train': 'cyan',
            'Val': 'yellow',
            'Test': 'green'
        }
        
        color = split_colors.get(split, 'white')
        header = f" {split} Metrics "
        print(colored(f"\n{'='*20} {header} {'='*20}", color, attrs=['bold']))
        
        if loss is not None:
            print(colored(f"Loss: {loss:.4f}", color))
        
        # Prepare table data with visual bars
        table_data = []
        headers = ["Task", "Accuracy", "F1 Score", "Precision", "Recall"]
        
        for task, metrics in metrics_dict.items():
            # Create visual bars for metrics (scaled to 20 chars)
            acc_bar = '█' * int(metrics['acc'] * 20)
            f1_bar = '█' * int(metrics['f1'] * 20)
            prec_bar = '█' * int(metrics['prec'] * 20)
            rec_bar = '█' * int(metrics['rec'] * 20)
            
            table_data.append([
                task,
                f"{metrics['acc']:.4f} {acc_bar}",
                f"{metrics['f1']:.4f} {f1_bar}",
                f"{metrics['prec']:.4f} {prec_bar}",
                f"{metrics['rec']:.4f} {rec_bar}"
            ])
        
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
        print(colored(f"{'='*50}", color))

    def _generate_visualizations(self, results_df):
        """
        Generate comprehensive visualizations for model evaluation.
        
        This method creates:
        1. Classification reports
        2. Confusion matrices for each task
        3. Performance comparison charts
        4. Misclassification examples
        
        Args:
            results_df (pd.DataFrame): DataFrame containing results
        """
        # Set up visualization style
        plt.style.use('fivethirtyeight')
        colors = ['#fc8d62', '#66c2a5', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494']
        sns.set(font_scale=1.2)
        sns.set_style("whitegrid", {'axes.grid': False})

        # Generate classification reports
        self._print_styled_report(classification_report(results_df['true_emotion'], results_df['pred_emotion']), 
                                "EMOTION CLASSIFICATION REPORT")
        self._print_styled_report(classification_report(results_df['true_sub_emotion'], results_df['pred_sub_emotion']), 
                                "SUB-EMOTION CLASSIFICATION REPORT")
        self._print_styled_report(classification_report(results_df['true_intensity'], results_df['pred_intensity']), 
                                "INTENSITY CLASSIFICATION REPORT")

        # Generate confusion matrices
        self._plot_enhanced_confusion_matrix(
            results_df['true_emotion'], results_df['pred_emotion'],
            self.emotion_encoder.classes_,
            'Emotion Confusion Matrix',
            cmap='YlGnBu'
        )

        self._plot_enhanced_confusion_matrix(
            results_df['true_intensity'], results_df['pred_intensity'],
            self.intensity_encoder.classes_,
            'Intensity Confusion Matrix',
            cmap='RdPu',
            figsize=(10, 8)
        )

        # Generate top sub-emotions confusion matrix
        top_sub_emotions = pd.Series(results_df['true_sub_emotion']).value_counts().nlargest(10).index.tolist()
        mask = np.isin(results_df['true_sub_emotion'], top_sub_emotions) & np.isin(results_df['pred_sub_emotion'], top_sub_emotions)
        self._plot_enhanced_confusion_matrix(
            results_df['true_sub_emotion'][mask],
            results_df['pred_sub_emotion'][mask],
            top_sub_emotions,
            'Top 10 Sub-Emotions Confusion Matrix',
            cmap='PuBuGn',
            figsize=(14, 12)
        )

        # Generate performance comparison charts
        self._plot_performance_comparison(results_df)

        # Show misclassified examples
        self._show_misclassified_examples(results_df)

    @staticmethod
    def _print_styled_report(report, title):
        """
        Print a styled classification report with color coding.
        
        Args:
            report (str): Classification report text
            title (str): Report title
        """
        report_lines = report.split('\n')
        print(f"\n{'='*80}")
        print(f"{title.center(80)}")
        print(f"{'='*80}")
        
        for line in report_lines:
            if not line.strip():
                continue
            if 'precision' in line or 'accuracy' in line:
                print(colored(line, 'cyan'))
            elif 'avg' in line:
                print(colored(line, 'yellow', attrs=['bold']))
            else:
                print(line)

    @staticmethod
    def _plot_enhanced_confusion_matrix(true_labels, pred_labels, classes, title, cmap='Blues', figsize=(12, 10)):
        """
        Plot an enhanced confusion matrix with normalized values and annotations.
        
        Args:
            true_labels (np.ndarray): True labels
            pred_labels (np.ndarray): Predicted labels
            classes (list): List of class names
            title (str): Plot title
            cmap (str): Color map for the heatmap
            figsize (tuple): Figure size (width, height)
        """
        # Get unique classes that actually appear in the data
        actual_classes = sorted(list(set(np.unique(true_labels)) | set(np.unique(pred_labels))))
        
        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create plot
        plt.figure(figsize=figsize)
        ax = plt.subplot()
        sns.heatmap(cm_norm, annot=cm, fmt='d', cmap=cmap, 
                    linewidths=0.5, cbar=True, square=True,
                    xticklabels=actual_classes, yticklabels=actual_classes,
                    annot_kws={"size": 10})
        
        f1 = f1_score(true_labels, pred_labels, average='weighted')
        plt.title(f"{title}\nF1 Score: {f1:.2%}", fontsize=16, fontweight='bold')

        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def _plot_performance_comparison(self, results_df):
        """
        Plot performance comparison charts for different tasks and metrics.
        
        Args:
            results_df (pd.DataFrame): DataFrame containing results
        """
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))

        # Plot emotion accuracy by class
        emotion_accuracy = {}
        for emotion in self.emotion_encoder.classes_:
            mask = results_df['true_emotion'] == emotion
            if mask.sum() > 0:
                emotion_accuracy[emotion] = np.mean(results_df.loc[mask, 'emotion_correct'])

        emotion_df = pd.DataFrame({'Accuracy': emotion_accuracy}).sort_values('Accuracy', ascending=False)
        sns.barplot(data=emotion_df, x=emotion_df.index, y='Accuracy', palette='viridis', ax=axes[0])
        axes[0].set_title('Accuracy by Emotion Category', fontsize=15)
        axes[0].set_ylim(0, 1)
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
        for i, v in enumerate(emotion_df['Accuracy']):
            axes[0].text(i, v + 0.02, f'{v:.2f}', ha='center')

        # Plot intensity accuracy by class
        intensity_accuracy = {}
        for intensity in self.intensity_encoder.classes_:
            mask = results_df['true_intensity'] == intensity
            if mask.sum() > 0:
                intensity_accuracy[intensity] = np.mean(results_df.loc[mask, 'intensity_correct'])

        intensity_df = pd.DataFrame({'Accuracy': intensity_accuracy}).sort_values('Accuracy', ascending=False)
        sns.barplot(data=intensity_df, x=intensity_df.index, y='Accuracy', palette='plasma', ax=axes[1])
        axes[1].set_title('Accuracy by Intensity Level', fontsize=15)
        axes[1].set_ylim(0, 1)
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
        for i, v in enumerate(intensity_df['Accuracy']):
            axes[1].text(i, v + 0.02, f'{v:.2f}', ha='center')

        # Plot overall metrics
        overall_metrics = {
            'Emotion': np.mean(results_df['emotion_correct']),
            'Sub_Emotion': np.mean(results_df['sub_emotion_correct']),
            'Intensity': np.mean(results_df['intensity_correct']),
            'All': np.mean(results_df['all_correct'])
        }

        overall_df = pd.DataFrame({'Task': list(overall_metrics.keys()), 'Accuracy': list(overall_metrics.values())})
        sns.barplot(data=overall_df, x='Task', y='Accuracy', palette='magma', ax=axes[2])
        axes[2].set_title('Overall Performance by Task', fontsize=15)
        axes[2].set_ylim(0, 1)
        for i, v in enumerate(overall_metrics.values()):
            axes[2].text(i, v + 0.02, f'{v:.2f}', ha='center')

        plt.tight_layout()
        plt.show()

    def _show_misclassified_examples(self, results_df):
        """
        Display examples of misclassified emotions for error analysis.
        
        Args:
            results_df (pd.DataFrame): DataFrame containing results
        """
        print("\n" + "="*80)
        print("MISCLASSIFICATION EXAMPLES".center(80))
        print("="*80)

        # Find most problematic emotion
        emotion_misclass = results_df[~results_df['emotion_correct']]
        most_problematic = emotion_misclass['true_emotion'].value_counts().idxmax()

        print(f"\nMost problematic emotion: {colored(most_problematic, 'red', attrs=['bold'])}")
        print(f"Examples of '{most_problematic}' misclassified:")

        # Show examples of misclassifications
        problematic_examples = emotion_misclass[emotion_misclass['true_emotion'] == most_problematic].sample(min(5, len(emotion_misclass)))
        for i, (_, row) in enumerate(problematic_examples.iterrows()):
            print(f"\n{i+1}. Text: {colored(row['text'][:100] + '...', 'cyan')}")
            print(f"   True: {colored(row['true_emotion'], 'green')} → Predicted: {colored(row['pred_emotion'], 'red')}")
            print(f"   Sub_emotion: {row['true_sub_emotion']} → {row['pred_sub_emotion']}")
            print(f"   Intensity: {row['true_intensity']} → {row['pred_intensity']}")