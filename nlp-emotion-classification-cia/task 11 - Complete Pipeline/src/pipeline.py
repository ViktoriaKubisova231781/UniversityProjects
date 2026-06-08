#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Emotion classification pipeline
-----------------------------------
This script implements a complete NLP pipeline that:
1. Downloads audio from YouTube
2. Transcribes audio to text
3. Translates text between languages
4. Classifies emotions in text
5. Saves results to file
"""

import os
import sys
import time
import warnings
import pandas as pd
import argparse  # Add argparse for command line arguments
from dotenv import load_dotenv

# Import the local modules
from data import *
from emotion_classifier import *
from speech_to_text import *
from translator import *

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Get paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")

# Initialize singleton instances for model caching (for faster performance)
# _emotion_predictor = EmotionPredictor()
# _pl_en_translator = PolishToEnglishTranslator(model_path=os.path.join(BASE_DIR, "models", "mt_pl_to_en"))
# _en_pl_translator = EnglishToPolishTranslator(model_path=os.path.join(BASE_DIR, "models", "mt_en_to_pl"))


def predict_emotion(texts, feature_config=None, reload_model=False):
    """
    Predict emotions for the given text(s).
    
    Args:
        texts (str or list): Text or list of texts to analyze
        feature_config (dict, optional): Configuration for features to use in prediction
        reload_model (bool): Force reload the model even if cached
        
    Returns:
        dict or list: Dictionary with emotion predictions for a single text or
                    list of dictionaries for multiple texts
    """
    _emotion_predictor = EmotionPredictor()
    start = time.time()
    try:
        output = _emotion_predictor.predict(texts, feature_config, reload_model)
        end = time.time()
        print(f"Latency (Emotion Classification): {end - start:.2f} seconds")
        return output
    except Exception as e:
        print(f"Error in emotion prediction: {str(e)}")
        return None


def speech_to_text(transcription_method, audio_file, output_file):
    """
    Perform speech-to-text transcription.
    
    Args:
        transcription_method (str): The method to use for transcription 
                                  ("assemblyAI" or "whisper")
        audio_file (str): Path to the input audio file
        output_file (str): Path for the output transcript file
    """
    start = time.time()
    try:
        if transcription_method.lower() == "assemblyai":
            api_key = os.environ.get("ASSEMBLYAI_API_KEY")
            if not api_key:
                raise ValueError("AssemblyAI API key not found in environment variables")
            transcriber = SpeechToTextTranscriber(api_key)
            transcriber.process(audio_file, output_file)
        elif transcription_method.lower() == "whisper":
            transcriber = WhisperTranscriber()
            transcriber.process(audio_file, output_file)
        else:
            raise ValueError(f"Unknown transcription method: {transcription_method}")
            
        end = time.time()
        print(f"Latency (Speech-to-Text): {end - start:.2f} seconds")
    except Exception as e:
        print(f"Error in speech-to-text: {str(e)}")


def translate(text, source_language, target_language):
    """
    Translate text from source language to target language.
    
    Args:
        text (str or list): Text or list of texts to translate
        source_language (str): Source language code (e.g., "en" for English)
        target_language (str): Target language code (e.g., "pl" for Polish)
        
    Returns:
        str or list: Translated text(s)
    """

    # Parameters for making the translation faster
    num_beams = 5   # Instead of 50 (default)
    temperature = 1.0  # Instead of 0.3 (default)

    start = time.time()
    try:
        if source_language == "en" and target_language == "pl":
            _en_pl_translator = EnglishToPolishTranslator(model_path=os.path.join(BASE_DIR, "models", "mt_en_to_pl"))
            result = _en_pl_translator.translate(text, num_beams=num_beams, temperature=temperature)
        elif source_language == "pl" and target_language == "en":
            _pl_en_translator = PolishToEnglishTranslator(model_path=os.path.join(BASE_DIR, "models", "mt_pl_to_en"))
            result = _pl_en_translator.translate(text, num_beams=num_beams, temperature=temperature)
        else:
            raise ValueError(f"Unsupported language pair: {source_language} to {target_language}")
            
        end = time.time()
        print(f"Latency (Translation): {end - start:.2f} seconds")
        return result
    except Exception as e:
        print(f"Error in translation: {str(e)}")
        return None


# Start the pipeline
if __name__ == "__main__":
    # Setup command-line arguments
    parser = argparse.ArgumentParser(description="Emotion Classification Pipeline")
    
    # YouTube URL argument
    parser.add_argument('--url', type=str, 
                      default="https://www.youtube.com/watch?v=ZDsfeIyjZUM",
                      help='YouTube video URL to download audio from')
    
    # Output filename argument
    parser.add_argument('--filename', type=str, 
                      default="top_models",
                      help='Filename for the downloaded audio file (without extension)')
    
    # Transcription method argument
    parser.add_argument('--transcription', type=str, 
                      choices=['assemblyAI', 'whisper'],
                      default="assemblyAI",
                      help='Method for speech-to-text transcription')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Store transcription method for later use
    transcription_method = args.transcription

    #########################################
    #    STEP 1 - DOWNLOAD YOUTUBE AUDIO    #
    #########################################

    # Report
    print("Step 1 - Downloading YouTube audio...")

    audio_file = os.path.join(BASE_DIR, "data", "youtube_audio", f"{args.filename}.mp3")    

    # Download the audio from a YouTube video
    audio_file_path = save_youtube_audio(
        url=args.url, 
        destination=os.path.join(BASE_DIR, "data", "youtube_audio"),
        return_path=True, 
        filename=args.filename
    )

    # Report
    print(f"Audio file saved at: {audio_file_path}")
    print("YouTube audio downloaded successfully!")


    ###############################################
    #    Step 2 - SPEECH TO TEXT TRANSCRIPTION    #
    ###############################################
    
    # Report
    print("Step 2 - Transcribing audio...")

    # Specify output file path for transcription
    output_file = os.path.join(BASE_DIR, "data", "transcription", f"transcribed_data_{transcription_method}.xlsx")

    # Transcribe the audio file
    speech_to_text(
        transcription_method, 
        audio_file, 
        output_file
    )

    # Load the transcribed data
    df = pd.read_excel(output_file)
    df = df.dropna(subset=["Sentence"])
    df = df.reset_index(drop=True)

    # Retrieve the sentences from the DataFrame
    sentences = df["Sentence"].tolist()

    # Report
    print(f"Transcription completed successfully!")


    ####################################
    #    STEP 3 - ROUND TRANSLATION    #
    ####################################

    # Report
    print("Step 3 - Translating sentences...")

    # Translate the sentences from English to Polish
    polish_translations = translate(sentences, source_language="en", target_language="pl")
    df["Polish Translation"] = polish_translations

    # Translate the sentences from Polish to English
    english_translations = translate(polish_translations, source_language="pl", target_language="en")
    df["English Translation"] = english_translations

    # Report
    print("Translations completed successfully!")


    #########################################
    #    STEP 4 - EMOTION CLASSIFICATION    #
    #########################################

    # Report
    print("Step 4 - Classifying emotions, sub-emotions, and intensity...")

    # Classify the sentences
    predictions = predict_emotion(sentences)

    # Add predictions to the DataFrame
    df["Emotion"] = [pred["emotion"] for pred in predictions]
    df["Sub Emotion"] = [pred["sub_emotion"] for pred in predictions]
    df["Intensity"] = [pred["intensity"] for pred in predictions]

    # Report
    print("Emotions classified successfully!")


    ###############################
    #    STEP 5 - SAVE RESULTS    #
    ###############################

    # Report
    print("Step 5 - Saving results...")

    # Save the results to an Excel file
    output_file = os.path.join(BASE_DIR, "data", "results", "results.xlsx")
    df.to_excel(output_file, index=False)

    # Report
    print(f"Results saved at: {output_file}")
    print("All steps completed successfully!")

