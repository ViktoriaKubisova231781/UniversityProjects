"""
Speech to Text Transcription using AssemblyAI API

This script takes an MP3 file as input and generates a structured output file containing
transcribed sentences. It uses AssemblyAI's API for high-accuracy transcription.

Requirements:
    - assemblyai package (install with: pip install -U assemblyai)
    - pandas package (install with: pip install pandas)
"""

import assemblyai as aai
import pandas as pd
import sys
import os
from typing import Optional

def setup_assemblyai(api_key: str) -> None:
    """Initialize AssemblyAI with the provided API key."""
    aai.settings.api_key = api_key

def transcribe_audio(file_path: str, config: Optional[aai.TranscriptionConfig] = None) -> aai.Transcript:
    """
    Transcribe the audio file using AssemblyAI.
    
    Args:
        file_path: Path to the audio file
        config: Optional transcription configuration
        
    Returns:
        AssemblyAI transcript object
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    transcriber = aai.Transcriber()
    
    try:
        transcript = transcriber.transcribe(file_path, config)
        
        if transcript.status == aai.TranscriptStatus.error:
            raise Exception(f"Transcription failed: {transcript.error}")
            
        return transcript
    
    except Exception as e:
        raise Exception(f"Transcription error: {str(e)}")

def save_transcript(transcript: aai.Transcript, output_file: str) -> None:
    """
    Save the transcript to a file (CSV/Excel) with sentences and timestamps.
    
    Args:
        transcript: AssemblyAI transcript object
        output_file: Path to save the output file
    """
    # Store sentences with their timestamps
    transcript_data = []
    
    try:
        # Use the sentences endpoint to get properly separated sentences
        for sentence in transcript.get_sentences():
            # Convert timestamps from milliseconds to seconds
            start_time = sentence.start / 1000 if sentence.start is not None else 0
            end_time = sentence.end / 1000 if sentence.end is not None else 0
            
            # Format timestamps as HH:MM:SS
            start_formatted = format_timestamp(start_time)
            end_formatted = format_timestamp(end_time)
            
            transcript_data.append({
                'Sentence': sentence.text,
                'Start Time': start_formatted,
                'End Time': end_formatted
            })
    except:
        # Fallback: Split the full text into sentences (less accurate)
        sentences = [{'Sentence': s.strip(), 
                     'Start Time': '', 
                     'End Time': ''} 
                    for s in transcript.text.split('.') if s.strip()]
        transcript_data = sentences
    
    # Create DataFrame with sentences and timestamps
    df = pd.DataFrame(transcript_data)
    
    # Save based on file extension
    file_ext = output_file.lower().split('.')[-1]
    if file_ext == 'csv':
        df.to_csv(output_file, index=False)
    elif file_ext in ['xlsx', 'xls']:
        df.to_excel(output_file, index=False)
    else:
        raise ValueError("Unsupported output file format. Use .csv or .xlsx")

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

def main(audio_file: str, output_file: str = "transcribed_data_assemblyAI.xlsx") -> None:
    """
    Main function to handle the transcription workflow.
    
    Args:
        audio_file: Path to the input audio file
        output_file: Path for the output transcript file
    """
    # AssemblyAI API key - Replace with your actual API key
    API_KEY = "*******"
    
    try:
        # Setup AssemblyAI
        setup_assemblyai(API_KEY)
        
        # Configure transcription
        config = aai.TranscriptionConfig(
            punctuate=True,  # Enable punctuation
            format_text=True  # Enable text formatting
        )
        
        # Perform transcription
        print(f"Transcribing {audio_file}...")
        transcript = transcribe_audio(audio_file, config)
        
        # Save results
        print(f"Saving transcript to {output_file}...")
        save_transcript(transcript, output_file)
        
        print("Transcription completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # Example usage
    audio_file = "./data_audio/ASIA'S NEXT TOP MODEL CYCLE 5 - EPISODE 1  ASIA BBG.mp3"
    main(audio_file)