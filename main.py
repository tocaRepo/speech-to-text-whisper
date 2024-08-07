from transformers import WhisperProcessor, WhisperForConditionalGeneration
import audio2numpy
import torchaudio
import numpy as np
import torch
from docx import Document
import datetime

def load_model_and_processor():
    """Load Whisper model and processor."""
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
    model.config.forced_decoder_ids = None
    return processor, model

def preprocess_audio(file_path):
    """Load and preprocess audio file."""
    input_speech, sr = audio2numpy.audio_from_file(file_path)
    resampled_input_speech = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(torch.Tensor(input_speech)).numpy()
    return resampled_input_speech

def transcribe_audio_chunks(audio_data, processor, model, chunk_size_seconds=60):
    """Transcribe audio data in chunks."""
    transcription = ""
    chunk_size_samples = chunk_size_seconds * 16000
    num_chunks = int(np.ceil(len(audio_data) / chunk_size_samples))

    for i in range(num_chunks):
        start_sample = i * chunk_size_samples
        end_sample = min((i + 1) * chunk_size_samples, len(audio_data))
        chunk = audio_data[start_sample:end_sample]
        input_features = processor(chunk, return_tensors="pt", sampling_rate=16000).input_features
        predicted_ids = model.generate(input_features, max_length=model.config.max_length, repetition_penalty=1)
        transcription_chunk = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        
        # Accumulate the transcription
        transcription += " " + " ".join(transcription_chunk)
        print(f"Transcription for chunk {i + 1}: {transcription_chunk}")

    return transcription

def save_transcription_to_docx(transcription):
    """Save transcription to a Word document."""
    doc = Document()
    doc.add_paragraph(transcription)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"transcription_{timestamp}.docx"
    doc.save(filename)
    print(f"Transcription saved to {filename}")

def main():
    print("CUDA Available:", torch.cuda.is_available())
    
    # Configuration
    audio_file_path = "audio/audio1.mp3"
    
    # Load model and processor
    processor, model = load_model_and_processor()
    
    # Preprocess audio
    audio_data = preprocess_audio(audio_file_path)
    
    # Transcribe audio
    transcription = transcribe_audio_chunks(audio_data, processor, model)
    
    # Save transcription
    save_transcription_to_docx(transcription)

if __name__ == "__main__":
    main()
