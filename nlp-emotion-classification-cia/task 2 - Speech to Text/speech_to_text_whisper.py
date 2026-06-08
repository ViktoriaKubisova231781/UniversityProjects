"""
Speech to Text Transcription using OpenAI's Whisper Model

This script takes an MP3 file as input and generates a structured output file containing
transcribed sentences with timestamps. It uses the Whisper model for high-accuracy transcription.

Requirements:
    - whisper package (install with: pip install -U openai-whisper)
    - pandas package (install with: pip install pandas)
    - ffmpeg (install with: apt-get install ffmpeg or brew install ffmpeg)
"""

import whisper
import pandas as pd
import sys
import os
from typing import Optional, List, Dict
import torch
import numpy as np

def load_whisper_model(model_size: str = "base") -> whisper.Whisper:
    """
    Load the Whisper model.
    
    Args:
        model_size: Size of the model to use ("tiny", "base", "small", "medium", "large")
        
    Returns:
        Loaded Whisper model
    """
    try:
        # Check if CUDA is available for GPU acceleration
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model(model_size).to(device)
        return model
    except Exception as e:
        raise Exception(f"Error loading Whisper model: {str(e)}")

def transcribe_audio(model: whisper.Whisper, 
                    file_path: str, 
                    language: Optional[str] = None) -> Dict:
    """
    Transcribe the audio file using Whisper.
    
    Args:
        model: Loaded Whisper model
        file_path: Path to the audio file
        language: Optional language code (e.g., "en" for English)
        
    Returns:
        Dictionary containing transcription results
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    try:
        # Transcribe with word-level timestamps
        result = model.transcribe(
            file_path,
            language=language,
            word_timestamps=True,
            verbose=False
        )
        return result
    
    except Exception as e:
        raise Exception(f"Transcription error: {str(e)}")

def format_timestamp(seconds: float) -> str:
    """
    Format time in seconds to HH:MM:SS format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string in HH:MM:SS format
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def extract_sentences(result: Dict) -> List[Dict]:
    """
    Extract sentences with timestamps from Whisper transcription result.
    
    Args:
        result: Whisper transcription result
        
    Returns:
        List of dictionaries containing sentences and their timestamps
    """
    transcript_data = []
    
    for segment in result['segments']:
        # Get the text and timestamps
        text = segment['text'].strip()
        start_time = segment['start']
        end_time = segment['end']
        
        # Format timestamps
        start_formatted = format_timestamp(start_time)
        end_formatted = format_timestamp(end_time)
        
        # Add to transcript data if there's text
        if text:
            transcript_data.append({
                'Sentence': text,
                'Start Time': start_formatted,
                'End Time': end_formatted
            })
    
    return transcript_data

def save_transcript(transcript_data: List[Dict], output_file: str) -> None:
    """
    Save the transcript to a file (CSV/Excel).
    
    Args:
        transcript_data: List of dictionaries containing transcription data
        output_file: Path to save the output file
    """
    # Create DataFrame
    df = pd.DataFrame(transcript_data)
    
    # Save based on file extension
    file_ext = output_file.lower().split('.')[-1]
    if file_ext == 'csv':
        df.to_csv(output_file, index=False)
    elif file_ext in ['xlsx', 'xls']:
        df.to_excel(output_file, index=False)
    else:
        raise ValueError("Unsupported output file format. Use .csv or .xlsx")

def main(audio_file: str, 
         output_file: str = "transcribed_data_whisper.xlsx",
         model_size: str = "base",
         language: Optional[str] = None) -> None:
    """
    Main function to handle the transcription workflow.
    
    Args:
        audio_file: Path to the input audio file
        output_file: Path for the output transcript file
        model_size: Size of the Whisper model to use
        language: Optional language code for transcription
    """
    try:
        # Load model
        print(f"Loading Whisper model ({model_size})...")
        model = load_whisper_model(model_size)
        
        # Perform transcription
        print(f"Transcribing {audio_file}...")
        result = transcribe_audio(model, audio_file, language)
        
        # Extract sentences with timestamps
        transcript_data = extract_sentences(result)
        
        # Save results
        print(f"Saving transcript to {output_file}...")
        save_transcript(transcript_data, output_file)
        
        print("Transcription completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # Example usage
    audio_file = "./data_audio/ASIA'S NEXT TOP MODEL CYCLE 5 - EPISODE 1  ASIA BBG.mp3"
    
    # You can specify different model sizes: "tiny", "base", "small", "medium", "large"
    # Larger models are more accurate but require more computational resources
    main(
        audio_file=audio_file,
        model_size="large",  # Change this to use different model sizes
        language="en"       # Specify language code or None for auto-detection
    )

