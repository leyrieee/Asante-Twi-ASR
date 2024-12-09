import streamlit as st
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa
import numpy as np
import tempfile
import os
from pydub import AudioSegment
import io

# Set page config
st.set_page_config(
    page_title="Asante Twi Speech Recognition",
    page_icon="🎙️",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
    .upload-text {
        text-align: center;
        padding: 2rem;
        border: 2px dashed #cccccc;
        border-radius: 0.5rem;
    }
    .title {
        text-align: center;
        color: #FF4B4B;
    }
    .result-box {
        padding: 1rem;
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    # Replace these with your model paths
    model = Wav2Vec2ForCTC.from_pretrained("your_model_path")
    processor = Wav2Vec2Processor.from_pretrained("your_processor_path")
    return model, processor

def convert_audio_to_wav(audio_file):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
        # Read the uploaded file
        audio_bytes = audio_file.read()
        
        # Get the format from the file extension
        file_format = audio_file.name.split('.')[-1].lower()
        
        # Create temporary input file
        with tempfile.NamedTemporaryFile(suffix=f'.{file_format}', delete=False) as temp_input:
            temp_input.write(audio_bytes)
            temp_input.flush()
            
            # Convert to WAV using pydub
            audio = AudioSegment.from_file(temp_input.name, format=file_format)
            audio = audio.set_frame_rate(16000)  # Set sample rate to 16kHz
            audio = audio.set_channels(1)  # Convert to mono
            audio.export(temp_wav.name, format='wav')
            
        # Clean up input temp file
        os.unlink(temp_input.name)
        
        return temp_wav.name

def process_audio(audio_path, model, processor):
    # Load and resample audio
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Process through model
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    return transcription

def main():
    # Header
    st.markdown("<h1 class='title'>🎙️ Asante Twi Speech Recognition</h1>", unsafe_allow_html=True)
    
    # Description
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        Transform your Asante Twi audio into text! Upload your audio file below.
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for layout
    col1, col2, col3 = st.columns([1,2,1])
    
    with col2:
        # File uploader with supported formats
        audio_file = st.file_uploader(
            "Upload your audio file",
            type=["wav", "mp3", "m4a", "ogg", "flac"],
            help="Supported formats: WAV, MP3, M4A, OGG, FLAC"
        )
    
    if audio_file:
        # Display file info
        st.markdown(
            f"<div style='text-align: center; color: #666;'>"
            f"📁 File: {audio_file.name}</div>",
            unsafe_allow_html=True
        )
        
        # Audio player
        st.audio(audio_file)
        
        # Transcribe button
        if st.button("🎯 Transcribe Audio"):
            with st.spinner("Processing your audio... This may take a moment ⏳"):
                try:
                    # Convert audio to WAV
                    wav_path = convert_audio_to_wav(audio_file)
                    
                    # Load model
                    model, processor = load_model()
                    
                    # Process audio and get transcription
                    transcription = process_audio(wav_path, model, processor)
                    
                    # Clean up temporary WAV file
                    os.unlink(wav_path)
                    
                    # Display results
                    st.markdown("### 📝 Transcription Result")
                    st.markdown(
                        f"<div class='result-box'>{transcription}</div>",
                        unsafe_allow_html=True
                    )
                    
                    # Add copy button
                    st.markdown(
                        f"""
                        <textarea id="transcription" style="position: absolute; left: -9999px;">{transcription}</textarea>
                        <button onclick="
                            navigator.clipboard.writeText(document.getElementById('transcription').value);
                            alert('Transcription copied to clipboard!');
                        ">📋 Copy to Clipboard</button>
                        """,
                        unsafe_allow_html=True
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error during transcription: {str(e)}")
                    st.error("Please try again with a different audio file.")

    # Footer
    st.markdown("""
        <div style='text-align: center; margin-top: 3rem; color: #666;'>
            <hr>
            <p>Powered by Wav2Vec2 Technology</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
