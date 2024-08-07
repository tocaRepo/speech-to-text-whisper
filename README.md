# speech-to-text-whisper - Whisper Audio Transcription

This project uses OpenAI's Whisper model to transcribe audio files and save the transcription to a Word document. The Whisper model is a powerful speech-to-text model that supports multiple languages and handles various audio qualities.

## Features

- **Audio Preprocessing**: Converts audio files to the required format and sample rate.
- **Chunk-based Transcription**: Handles long audio files by processing them in chunks.
- **Document Generation**: Saves the transcription to a Word document with a timestamp.

## Requirements

To run this project, you need Python 3.7 or higher and the following Python packages:

- `transformers`
- `torchaudio`
- `audio2numpy`
- `numpy`
- `torch`
- `python-docx`


## Usage
Prepare Your Audio File: Place your audio file (e.g., MP3) in the audio/ directory. You can adjust the file path in the main function of the main.py script if needed.

Run the Script: Execute the main.py script to start the transcription process.

Check Transcription: The script will output the transcription of each audio chunk to the console and save the complete transcription to a Word document in the current directory.


Ensure your audio file is in a supported format and properly placed in the audio/ directory.
The script assumes a sample rate of 16,000 Hz for the Whisper model.