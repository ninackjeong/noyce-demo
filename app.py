import streamlit as st
import numpy as np
import torch
import librosa
import matplotlib.pyplot as plt
from transformers import pipeline, AutoModelForAudioClassification

# Page configuration
st.set_page_config(
    page_title="Noyce Demo: Emotion Recognition",
    page_icon="🎵",
    layout="wide"
)

# Cache the model loading to improve performance
@st.cache_resource
def load_emotion_model():
    # Using a model fine-tuned for speech emotion recognition
    model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    
    # Use pipeline instead of separate processor and model
    emotion_classifier = pipeline("audio-classification", model=model_name)
    
    # Get the model for additional information if needed
    model = emotion_classifier.model
    
    return emotion_classifier, model

# Load models
emotion_classifier, model = load_emotion_model()

# App title and description
st.title("Noyce Demo: Emotion Recognition")
st.markdown("Upload an audio file to detect emotions from speech.")

# Function to preprocess audio
def preprocess_audio(audio_file, target_sr=16000):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=None)
    
    # Resample if needed
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    
    return y, target_sr

# Function to predict emotion from audio
def predict_emotion(audio_data, sampling_rate, classifier):
    # Use the pipeline directly for prediction
    predictions = classifier({"raw": audio_data, "sampling_rate": sampling_rate})
    
    # Get predicted emotion and confidence
    emotion = predictions[0]["label"]
    confidence = predictions[0]["score"]
    
    # Get all emotion probabilities
    all_emotions = {pred["label"]: pred["score"] for pred in predictions}
    
    return emotion, confidence, all_emotions

# Function to plot audio waveform
def plot_waveform(audio_data, sr):
    fig, ax = plt.subplots(figsize=(10, 2))
    librosa.display.waveshow(audio_data, sr=sr, ax=ax)
    ax.set_title("Audio Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    return fig

# Function to plot spectrogram
def plot_spectrogram(audio_data, sr):
    fig, ax = plt.subplots(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax)
    ax.set_title("Spectrogram")
    fig.colorbar(ax.collections[0], ax=ax, format="%+2.f dB")
    return fig

# Function to plot emotion probabilities
def plot_emotion_probs(emotions_dict):
    # Sort emotions by probability
    sorted_emotions = sorted(emotions_dict.items(), key=lambda x: x[1], reverse=True)
    labels, values = zip(*sorted_emotions)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color='skyblue')
    
    # Highlight the highest value
    bars[0].set_color('navy')
    
    ax.set_title("Emotion Probabilities")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.0)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

# Main app functionality
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # Display audio player
    st.audio(uploaded_file, format="audio/wav")
    
    # Process audio
    with st.spinner("Analyzing audio..."):
        # Preprocess the audio
        audio_data, sr = preprocess_audio(uploaded_file)
        
        # Make prediction
        emotion, confidence, all_emotions = predict_emotion(audio_data, sr, emotion_classifier)
    
    # Display results
    st.success("Analysis complete!")
    
    # Create columns for results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Detected Emotion")
        st.markdown(f"### {emotion.capitalize()} ({confidence:.2f})")
        
        # Map emotions to emojis
        emoji_map = {
            "angry": "😠", "disgust": "🤢", "fear": "😨", 
            "happy": "😃", "neutral": "😐", "sad": "😢", 
            "surprise": "😲", "calm": "😌", "excited": "🤩",
            "frustrated": "😤", "other": "❓"
        }
        
        emoji = emoji_map.get(emotion.lower(), "❓")
        st.markdown(f"## {emoji}")
        
        # Display audio waveform
        st.subheader("Waveform")
        waveform_fig = plot_waveform(audio_data, sr)
        st.pyplot(waveform_fig)
    
    with col2:
        # Display all emotions probabilities
        st.subheader("Emotion Probabilities")
        probs_fig = plot_emotion_probs(all_emotions)
        st.pyplot(probs_fig)
        
        # Display spectrogram
        st.subheader("Spectrogram")
        spec_fig = plot_spectrogram(audio_data, sr)
        st.pyplot(spec_fig)
    
    # Add audio details
    st.subheader("Audio Details")
    st.write(f"Duration: {len(audio_data)/sr:.2f} seconds")
    st.write(f"Sampling Rate: {sr} Hz")

# Add example section
st.markdown("---")
st.subheader("Example Inputs and Expected Outputs")

# Create tabs for examples
tab1, tab2, tab3 = st.tabs(["Happy Example", "Angry Example", "Sad Example"])

with tab1:
    st.write("Example of a **happy** audio sample:")
    st.write("- Input: Audio clip of someone speaking with excitement and joy")
    st.write("- Expected output: Primary emotion 'happy' with high confidence")
    st.write("- Secondary emotions might include 'excited' or 'surprise'")
    st.markdown("""
    ```
    Example Output:
    Detected Emotion: Happy (0.87)
    Audio Duration: 3.45 seconds
    ```
    """)

with tab2:
    st.write("Example of an **angry** audio sample:")
    st.write("- Input: Audio clip of someone speaking in an angry tone")
    st.write("- Expected output: Primary emotion 'angry' with high confidence")
    st.write("- Secondary emotions might include 'frustrated' or 'disgust'")
    st.markdown("""
    ```
    Example Output:
    Detected Emotion: Angry (0.92)
    Audio Duration: 2.78 seconds
    ```
    """)

with tab3:
    st.write("Example of a **sad** audio sample:")
    st.write("- Input: Audio clip of someone speaking in a sad tone")
    st.write("- Expected output: Primary emotion 'sad' with high confidence")
    st.write("- Secondary emotions might include 'neutral' or 'fear'")
    st.markdown("""
    ```
    Example Output:
    Detected Emotion: Sad (0.76)
    Audio Duration: 4.12 seconds
    ```
    """)

# App information
st.sidebar.title("About")
st.sidebar.info(
    "This app uses a pre-trained model from Hugging Face to detect emotions in speech audio. "
    "Upload an audio file (WAV, MP3, OGG) to analyze the emotional content."
)

st.sidebar.title("Sample Data")
st.sidebar.markdown("""
You can find sample emotional audio datasets at:
- [RAVDESS Dataset](https://zenodo.org/record/1188976)
- [CREMA-D Dataset](https://github.com/CheyneyComputerScience/CREMA-D)
- [TESS Dataset](https://tspace.library.utoronto.ca/handle/1807/24487)
""")

st.sidebar.title("Tips")
st.sidebar.markdown("""
For best results:
- Use clear audio with minimal background noise
- Audio should contain speech (not just music)
- Short clips (2-10 seconds) work best
- The model works best with English speech
""")