"""
Text augmentation module for emotion classification datasets.
This module provides various text augmentation techniques to handle class imbalance
and improve model generalization for emotion classification tasks.
"""

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import wordnet, stopwords
import random
import re
import contractions
from collections import Counter
import logging
from tqdm import tqdm

# Download required NLTK resources
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)

class TextAugmentor:
    """
    Text augmentation class for emotion classification datasets.
    
    This class provides methods to augment text data for addressing class imbalance
    and improving model generalization through various text transformation techniques.
    
    Attributes:
        random_state (int): Random seed for reproducibility
        stop_words (set): Set of English stopwords
        augmentation_methods (dict): Dictionary of available augmentation methods
        emotion_specific_words (dict): Dictionary of emotion-specific words for targeted augmentation
    """
    
    def __init__(self, random_state=42):
        """
        Initialize TextAugmentor with configuration parameters.
        
        Args:
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        # Set random seed for reproducibility
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)
        
        # Initialize stopwords
        self.stop_words = set(stopwords.words('english'))
        
        # Dictionary of available augmentation methods
        self.augmentation_methods = {
            'synonym_replacement': self.synonym_replacement,
            'random_deletion': self.random_deletion,
            'random_swap': self.random_swap,
            'random_insertion': self.random_insertion,
            'text_noise': self.text_noise,
            'contractions_expansion': self.contractions_expansion,
            'emoji_insertion': self.emoji_insertion,
        }
        
        # Dictionary of emotion-specific intensifiers and keywords
        self.emotion_specific_words = {
            'happiness': ['happy', 'joyful', 'excited', 'thrilled', 'delighted', 'pleased', 'glad', 'content'],
            'sadness': ['sad', 'upset', 'depressed', 'heartbroken', 'gloomy', 'miserable', 'somber', 'unhappy'],
            'anger': ['angry', 'furious', 'outraged', 'annoyed', 'irritated', 'enraged', 'mad', 'frustrated'],
            'fear': ['afraid', 'scared', 'terrified', 'frightened', 'anxious', 'paranoid', 'nervous', 'worried'],
            'love': ['love', 'affection', 'adore', 'cherish', 'fond', 'attachment', 'devotion', 'passion'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'startled', 'stunned', 'speechless', 'awestruck'],
            'neutral': ['okay', 'fine', 'alright', 'whatever', 'indifferent', 'unaffected', 'unbothered', 'casual'],
            # More specific emotions
            'curiosity': ['curious', 'wonder', 'intrigued', 'interested', 'questioning', 'suspicious', 'puzzled'],
            'annoyance': ['annoyed', 'irritated', 'bothered', 'frustrated', 'aggravated', 'displeased', 'irked'],
            'confusion': ['confused', 'puzzled', 'perplexed', 'baffled', 'bewildered', 'disoriented', 'uncertain'],
            'pride': ['proud', 'accomplished', 'confident', 'dignified', 'satisfied', 'self-respecting', 'honored'],
            'optimism': ['optimistic', 'hopeful', 'positive', 'confident', 'encouraged', 'upbeat', 'looking forward'],
            'realization': ['realized', 'understood', 'recognized', 'comprehended', 'grasped', 'discovered', 'acknowledged']
        }
        
        # Emoji dictionary for emotion-based augmentation
        self.emoji_dict = {
            'happiness': ['😊', '😃', '😄', '😁', '🙂', '😀', '😋', '🥰'],
            'sadness': ['😢', '😭', '😔', '😞', '😥', '😪', '😓', '🥺'],
            'anger': ['😠', '😡', '🤬', '😤', '😒', '👿', '💢', '💥'],
            'fear': ['😨', '😱', '😰', '😯', '😦', '🙀', '😳', '🤯'],
            'love': ['❤️', '😍', '🥰', '💕', '💞', '💓', '💗', '💖'],
            'surprise': ['😮', '😲', '😯', '😦', '😧', '😵', '🤔', '🙀'],
            'neutral': ['😐', '😑', '😶', '🙄', '😏', '😌', '😕', '😬'],
            'curiosity': ['🤔', '🧐', '❓', '🔍', '👀', '👂', '👁️', '❔'],
            'annoyance': ['😒', '😑', '🙄', '😣', '😫', '😤', '🤨', '😬'],
            'confusion': ['😕', '😵', '🤔', '😯', '❓', '🧐', '😟', '😦'],
            'pride': ['😎', '🦚', '🦁', '🏆', '🥇', '👑', '🤴', '👸'],
            'optimism': ['🌈', '🌞', '✨', '🔆', '☀️', '🚀', '💫', '⭐'],
            'realization': ['💡', '👀', '‼️', '❗', '⚡', '🤯', '🧠', '✅']
        }
        
    def _get_synonyms(self, word):
        """
        Get synonyms for a word using WordNet.
        
        Args:
            word (str): Input word to find synonyms for
            
        Returns:
            list: List of synonyms
        """
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym != word and synonym not in synonyms:
                    synonyms.append(synonym)
        return synonyms
    
    def synonym_replacement(self, text, n=1):
        """
        Replace n random words in the text with their synonyms.
        
        Args:
            text (str): Input text
            n (int, optional): Number of words to replace. Defaults to 1.
            
        Returns:
            str: Augmented text with synonym replacements
        """
        words = text.split()
        if len(words) <= 1:
            return text
            
        # Filter words that are not stopwords and have synonyms
        candidates = []
        for i, word in enumerate(words):
            if word.lower() not in self.stop_words and len(word) > 3:
                synonyms = self._get_synonyms(word)
                if synonyms:
                    candidates.append((i, word, synonyms))
        
        # If no viable candidates, return original text
        if not candidates:
            return text
            
        # Perform replacements
        n = min(n, len(candidates))
        chosen = random.sample(candidates, n)
        
        for i, original, synonyms in chosen:
            words[i] = random.choice(synonyms)
            
        return ' '.join(words)
    
    def random_deletion(self, text, p=0.1):
        """
        Randomly delete words from the text with probability p.
        
        Args:
            text (str): Input text
            p (float, optional): Probability of deletion. Defaults to 0.1.
            
        Returns:
            str: Augmented text with random deletions
        """
        words = text.split()
        if len(words) <= 3:  # Don't delete if text is too short
            return text
            
        # Keep words with probability (1-p)
        kept_words = []
        for word in words:
            if word.lower() in self.stop_words or random.random() > p:
                kept_words.append(word)
                
        # If all words were deleted, keep a random one
        if not kept_words:
            kept_words = [random.choice(words)]
            
        return ' '.join(kept_words)
    
    def random_swap(self, text, n=1):
        """
        Randomly swap the positions of two words in the text n times.
        
        Args:
            text (str): Input text
            n (int, optional): Number of swaps to perform. Defaults to 1.
            
        Returns:
            str: Augmented text with random swaps
        """
        words = text.split()
        if len(words) <= 1:
            return text
            
        for _ in range(n):
            # Get two random indices
            idx1, idx2 = random.sample(range(len(words)), 2)
            # Swap words
            words[idx1], words[idx2] = words[idx2], words[idx1]
            
        return ' '.join(words)
    
    def random_insertion(self, text, n=1):
        """
        Insert n synonyms into random positions in the text.
        
        Args:
            text (str): Input text
            n (int, optional): Number of insertions to perform. Defaults to 1.
            
        Returns:
            str: Augmented text with random insertions
        """
        words = text.split()
        if len(words) == 0:
            return text
            
        # Get candidate words for finding synonyms
        candidates = [word for word in words if word.lower() not in self.stop_words and len(word) > 3]
        if not candidates:
            return text
            
        for _ in range(n):
            # Choose a random word and get its synonyms
            insert_word = random.choice(candidates)
            synonyms = self._get_synonyms(insert_word)
            
            # If synonyms found, insert one at a random position
            if synonyms:
                synonym = random.choice(synonyms)
                insert_pos = random.randint(0, len(words))
                words.insert(insert_pos, synonym)
                
        return ' '.join(words)
    
    def text_noise(self, text, p=0.05):
        """
        Add random character-level noise to the text.
        
        Args:
            text (str): Input text
            p (float, optional): Probability of noise. Defaults to 0.05.
            
        Returns:
            str: Augmented text with character-level noise
        """
        chars = list(text)
        
        for i in range(len(chars)):
            if random.random() < p:
                # Choose a noise type
                noise_type = random.choice(['insert', 'delete', 'swap', 'replace'])
                
                if noise_type == 'insert' and i < len(chars) - 1:
                    # Insert a random character
                    chars.insert(i, random.choice('abcdefghijklmnopqrstuvwxyz '))
                elif noise_type == 'delete' and len(chars) > 3:
                    # Delete this character
                    chars.pop(i)
                    break  # Break to avoid index error
                elif noise_type == 'swap' and i < len(chars) - 1:
                    # Swap with next character
                    chars[i], chars[i+1] = chars[i+1], chars[i]
                elif noise_type == 'replace':
                    # Replace with a random character
                    chars[i] = random.choice('abcdefghijklmnopqrstuvwxyz ')
        
        return ''.join(chars)
    
    def contractions_expansion(self, text):
        """
        Expand or contract words in the text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Augmented text with expanded or contracted words
        """
        # Find all contractions in the text
        found_contractions = re.findall(r'\b\w+\'\w+\b', text)
        
        if found_contractions and random.random() < 0.5:
            # Expand contractions
            return contractions.fix(text)
        else:
            # Contract expanded forms
            words = text.split()
            if len(words) >= 2:
                for i in range(len(words) - 1):
                    pair = ' '.join(words[i:i+2])
                    
                    # Common expansions to contract
                    expansions = {
                        'i am': "I'm", 'do not': "don't", 'cannot': "can't", 
                        'will not': "won't", 'is not': "isn't", 'are not': "aren't",
                        'was not': "wasn't", 'were not': "weren't", 'have not': "haven't",
                        'has not': "hasn't", 'had not': "hadn't", 'would not': "wouldn't",
                        'should not': "shouldn't", 'could not': "couldn't"
                    }
                    
                    if pair.lower() in expansions:
                        # Replace the pair with its contraction
                        replacement = expansions[pair.lower()]
                        words[i] = replacement
                        words.pop(i+1)
                        
            return ' '.join(words)
    
    def emoji_insertion(self, text, emotion=None, p=0.7):
        """
        Insert emotion-relevant emojis to the text.
        
        Args:
            text (str): Input text
            emotion (str, optional): Emotion label. If provided, relevant emojis will be used.
            p (float, optional): Probability of inserting emoji. Defaults to 0.7.
            
        Returns:
            str: Augmented text with emojis
        """
        if random.random() > p:
            return text
            
        # Get relevant emojis
        if emotion and emotion in self.emoji_dict:
            emojis = self.emoji_dict[emotion]
        else:
            # Use general positive/neutral emojis if emotion not specified
            emojis = self.emoji_dict['neutral']
            
        # Insert 1-2 emojis at beginning, end, or both
        position = random.choice(['start', 'end', 'both'])
        result = text
        
        if position in ['start', 'both']:
            emoji = random.choice(emojis)
            result = f"{emoji} {result}"
            
        if position in ['end', 'both']:
            emoji = random.choice(emojis)
            result = f"{result} {emoji}"
            
        return result
    
    def emotion_specific_augmentation(self, text, emotion):
        """
        Apply emotion-specific augmentations to enhance emotional content.
        
        Args:
            text (str): Input text
            emotion (str): Emotion label
            
        Returns:
            str: Augmented text with emotion-specific enhancements
        """
        if not emotion or emotion not in self.emotion_specific_words:
            return text
            
        words = text.split()
        
        # 40% chance of adding emotion-specific words/intensifiers
        if random.random() < 0.4 and len(words) > 2:
            # Get emotion-specific keywords
            emotion_words = self.emotion_specific_words[emotion]
            
            # Choose augmentation strategy
            strategy = random.choice(['insert', 'replace', 'intensify'])
            
            if strategy == 'insert' and emotion_words:
                # Insert emotion keyword at random position
                insert_pos = random.randint(0, len(words))
                insert_word = random.choice(emotion_words)
                words.insert(insert_pos, insert_word)
                
            elif strategy == 'replace' and emotion_words:
                # Replace a non-stop word with emotion keyword
                replace_candidates = [i for i, word in enumerate(words) 
                                    if word.lower() not in self.stop_words]
                if replace_candidates:
                    replace_pos = random.choice(replace_candidates)
                    words[replace_pos] = random.choice(emotion_words)
                    
            elif strategy == 'intensify':
                # Add intensifiers based on emotion
                intensifiers = {
                    'happiness': ['really', 'extremely', 'so', 'absolutely'],
                    'sadness': ['deeply', 'terribly', 'incredibly', 'utterly'],
                    'anger': ['absolutely', 'completely', 'totally', 'incredibly'],
                    'fear': ['extremely', 'absolutely', 'completely', 'utterly'],
                    'love': ['deeply', 'truly', 'absolutely', 'completely'],
                    'surprise': ['totally', 'completely', 'absolutely', 'utterly'],
                    'neutral': ['quite', 'rather', 'fairly', 'somewhat'],
                    'curiosity': ['really', 'quite', 'genuinely', 'truly'],
                    'annoyance': ['extremely', 'rather', 'fairly', 'very'],
                    'confusion': ['completely', 'utterly', 'totally', 'absolutely'],
                    'pride': ['extremely', 'genuinely', 'truly', 'deeply'],
                    'optimism': ['truly', 'genuinely', 'really', 'very'],
                    'realization': ['suddenly', 'completely', 'totally', 'finally']
                }
                
                if emotion in intensifiers:
                    insert_pos = random.randint(0, len(words))
                    intensifier = random.choice(intensifiers[emotion])
                    words.insert(insert_pos, intensifier)
                
        return ' '.join(words)
    
    def augment_text(self, text, emotion=None, methods=None, n_methods=1, debug=False):
        """
        Apply multiple augmentation methods to a text.
        
        Args:
            text (str): Input text
            emotion (str, optional): Emotion label. Defaults to None.
            methods (list, optional): List of methods to apply. If None, randomly select.
            n_methods (int, optional): Number of methods to apply. Defaults to 1.
            debug (bool, optional): Whether to print debugging info. Defaults to False.
            
        Returns:
            str: Augmented text
        """
        if not text or len(text.strip()) == 0:
            return text
            
        # Select methods to apply
        available_methods = list(self.augmentation_methods.keys())
        if methods:
            methods_to_apply = [m for m in methods if m in available_methods]
        else:
            # Randomly select n_methods
            n = min(n_methods, len(available_methods))
            methods_to_apply = random.sample(available_methods, n)
            
        # Apply selected methods
        augmented = text
        applied_methods = []
        
        for method_name in methods_to_apply:
            method = self.augmentation_methods[method_name]
            
            # Apply method with emotion parameter if it's emoji_insertion
            if method_name == 'emoji_insertion' and emotion:
                augmented = method(augmented, emotion)
            else:
                augmented = method(augmented)
                
            applied_methods.append(method_name)
            
        # Apply emotion-specific augmentation as a final step
        if emotion:
            augmented = self.emotion_specific_augmentation(augmented, emotion)
            applied_methods.append('emotion_specific')
            
        if debug:
            print(f"Original: {text}")
            print(f"Augmented: {augmented}")
            print(f"Applied methods: {applied_methods}")
            
        return augmented
    
    def balance_dataset(self, df, text_column='text', emotion_column='sub_emotion', 
                       target_count=None, augmentation_ratio=2, random_state=None):
        """
        Balance dataset by augmenting minority classes.
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str, optional): Column containing text. Defaults to 'text'.
            emotion_column (str, optional): Column containing emotion labels. Defaults to 'sub_emotion'.
            target_count (int, optional): Target count for each class. If None, use majority class count.
            augmentation_ratio (int, optional): Max ratio of augmented to original samples. Defaults to 2.
            random_state (int, optional): Random seed. Defaults to None.
            
        Returns:
            pd.DataFrame: Balanced dataframe with augmented samples
        """
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
            
        # Count samples per class
        class_counts = df[emotion_column].value_counts()
        
        # Determine target count
        if target_count is None:
            target_count = class_counts.max()
        
        # Create new dataframe for augmented samples
        augmented_rows = []
        
        # Process each class
        for emotion, count in tqdm(class_counts.items(), desc="Balancing classes"):
            # Skip if already at or above target
            if count >= target_count:
                continue
                
            # Get samples for this class
            class_samples = df[df[emotion_column] == emotion]
            
            # Determine number of augmentations needed
            n_to_generate = min(target_count - count, count * augmentation_ratio)
            
            # Generate augmented samples
            augmentation_methods = list(self.augmentation_methods.keys())
            
            for _ in range(n_to_generate):
                # Randomly select a sample to augment
                sample = class_samples.sample(1).iloc[0]
                
                # Randomly select 1-3 augmentation methods
                n_methods = random.randint(1, 3)
                methods = random.sample(augmentation_methods, n_methods)
                
                # Augment the text
                augmented_text = self.augment_text(
                    sample[text_column], 
                    emotion=emotion,
                    methods=methods
                )
                
                # Create new row with augmented text
                new_row = sample.copy()
                new_row[text_column] = augmented_text
                
                # Add augmentation marker if needed
                if 'augmented' not in new_row:
                    new_row['augmented'] = True
                    
                augmented_rows.append(new_row)
        
        # Create augmented dataframe
        augmented_df = pd.DataFrame(augmented_rows)
        
        # Combine original and augmented dataframes
        balanced_df = pd.concat([df, augmented_df], ignore_index=True)
        
        # Add augmentation marker to original samples if needed
        if 'augmented' in balanced_df.columns:
            balanced_df.loc[balanced_df['augmented'].isna(), 'augmented'] = False
            
        # Shuffle the final dataframe
        balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Print statistics
        print(f"Original dataset: {len(df)} samples")
        print(f"Augmented samples: {len(augmented_rows)} samples")
        print(f"Balanced dataset: {len(balanced_df)} samples")
        print("\nClass distribution:")
        print(balanced_df[emotion_column].value_counts())
        
        return balanced_df
    
    def generate_equal_samples(self, df, text_column='text', emotion_column='sub_emotion', 
                             samples_per_class=500, random_state=None):
        """
        Generate exactly equal number of samples per class through augmentation.
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str, optional): Column containing text. Defaults to 'text'.
            emotion_column (str, optional): Column containing emotion labels. Defaults to 'sub_emotion'.
            samples_per_class (int, optional): Number of samples per class. Defaults to 500.
            random_state (int, optional): Random seed. Defaults to None.
            
        Returns:
            pd.DataFrame: DataFrame with equal samples per class
        """
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
            
        # Get unique classes
        classes = df[emotion_column].unique()
        
        # Create new dataframe for equal samples
        equal_samples = []
        
        for emotion in tqdm(classes, desc="Generating equal samples"):
            # Get samples for this class
            class_samples = df[df[emotion_column] == emotion]
            count = len(class_samples)
            
            # If we have enough or more, just sample randomly
            if count >= samples_per_class:
                equal_samples.append(class_samples.sample(samples_per_class, random_state=random_state))
                continue
                
            # If we need more samples, augment existing ones
            augmented_rows = []
            
            # First add all existing samples
            augmented_rows.append(class_samples)
            
            # Then generate the rest through augmentation
            samples_to_generate = samples_per_class - count
            
            # We'll need to augment some samples multiple times
            augmentation_rounds = (samples_to_generate + count - 1) // count
            
            for round_idx in range(augmentation_rounds):
                # For each round, augment all original samples
                for _, sample in class_samples.iterrows():
                    if len(augmented_rows) + 1 > samples_per_class:
                        break
                        
                    # Use more aggressive augmentation for later rounds
                    n_methods = min(1 + round_idx, 3)
                    
                    augmented_text = self.augment_text(
                        sample[text_column],
                        emotion=emotion,
                        n_methods=n_methods
                    )
                    
                    # Create new row with augmented text
                    new_row = sample.copy()
                    new_row[text_column] = augmented_text
                    new_row['augmented'] = True
                    
                    augmented_rows.append(pd.DataFrame([new_row]))
            
            # Combine all samples for this class
            class_combined = pd.concat(augmented_rows, ignore_index=True)
            
            # If we generated too many, sample randomly
            if len(class_combined) > samples_per_class:
                class_combined = class_combined.sample(samples_per_class, random_state=random_state)
                
            equal_samples.append(class_combined)
        
        # Combine all classes
        balanced_df = pd.concat(equal_samples, ignore_index=True)
        
        # Add augmentation marker to original samples if needed
        if 'augmented' in balanced_df.columns:
            balanced_df.loc[balanced_df['augmented'].isna(), 'augmented'] = False
            
        # Shuffle the final dataframe
        balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Print statistics
        print(f"Original dataset: {len(df)} samples")
        print(f"Balanced dataset: {len(balanced_df)} samples")
        print(f"Samples per class: {samples_per_class}")
        print("\nClass distribution:")
        print(balanced_df[emotion_column].value_counts())
        
        return balanced_df 
    

"""
Example script demonstrating the use of the TextAugmentor class
for balancing the emotion dataset for DeBERTa model training.


import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Add helper directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helper.augmentation import TextAugmentor

def main():
    # Load a sample of the dataset (replace with actual data loading code)
    # This is just an example, replace with your actual dataframe
    df = pd.DataFrame({
        'text': [
            'Did you kiss?',
            'Yeah.',
            'But no !',
            'Why do you have a problem?',
            'So, you wanted a girl like that?',
            'And tribes put everything on the line.',
            'I came in claiming that I had the most knowledge.',
            'At no point during my time here was that disproven.',
            'There is nothing in the last few days of my experience that I regret.',
            'I think that looking back and overthinking it is pointless.'
        ],
        'sub_emotion': [
            'curiosity',
            'neutral',
            'annoyance',
            'confusion',
            'curiosity',
            'optimism',
            'pride',
            'pride',
            'pride',
            'realization'
        ]
    })
    
    print("Original dataset:")
    print(df.head())
    print("\nClass distribution:")
    print(df['sub_emotion'].value_counts())
    
    # Create TextAugmentor instance
    augmentor = TextAugmentor(random_state=42)
    
    # Example 1: Show individual augmentation examples
    print("\n--- Example Augmentations ---")
    for method in augmentor.augmentation_methods.keys():
        text = df['text'].iloc[0]  # Use first text as example
        emotion = df['sub_emotion'].iloc[0]
        augmented = augmentor.augmentation_methods[method](text)
        print(f"{method}:")
        print(f"  Original: {text}")
        print(f"  Augmented: {augmented}")
    
    # Example 2: Balance dataset using majority class as target
    print("\n--- Balanced Dataset (to majority class) ---")
    balanced_df = augmentor.balance_dataset(
        df, 
        text_column='text',
        emotion_column='sub_emotion',
        target_count=None,  # Use majority class as target
        augmentation_ratio=2
    )
    
    # Visualize before and after class distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.countplot(data=df, x='sub_emotion')
    plt.title('Original Class Distribution')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    sns.countplot(data=balanced_df, x='sub_emotion')
    plt.title('Balanced Class Distribution')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('class_balance_comparison.png')
    plt.close()
    
    # Example 3: Create dataset with exactly equal class sizes
    print("\n--- Equal Samples Per Class ---")
    equal_df = augmentor.generate_equal_samples(
        df,
        text_column='text',
        emotion_column='sub_emotion',
        samples_per_class=3  # Generate exactly 3 samples per class
    )
    
    # Show examples of augmented text for one class
    print("\n--- Augmented Samples (Pride emotion) ---")
    pride_samples = equal_df[equal_df['sub_emotion'] == 'pride']
    for i, row in pride_samples.iterrows():
        print(f"Sample {i}:")
        print(f"Text: {row['text']}")
        print(f"Augmented: {row.get('augmented', 'No')}")
        print()
    
    # Show examples of combined augmentation techniques
    print("\n--- Combined Augmentation Examples ---")
    for i in range(5):
        text = df['text'].iloc[i % len(df)]
        emotion = df['sub_emotion'].iloc[i % len(df)]
        augmented = augmentor.augment_text(
            text, 
            emotion=emotion,
            n_methods=2,
            debug=True
        )

if __name__ == "__main__":
    main()
    print("\nAugmentation examples completed. Check class_balance_comparison.png for visualization.") 

"""