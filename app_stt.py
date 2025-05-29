import streamlit as st
import numpy as np
import torch
import librosa
import matplotlib.pyplot as plt
from transformers import pipeline, AutoProcessor, AutoModelForAudioClassification

# Page configuration
st.set_page_config(
    page_title="Audio Emotion Recognition",
    page_icon="🎵",
    layout="wide"
)

# Cache the model loading to improve performance
@st.cache_resource
def load_emotion_model():
    # Using a model fine-tuned for speech emotion recognition
    model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForAudioClassification.from_pretrained(model_name)
    return processor, model

# Load models
processor, model = load_emotion_model()

# Cache the STT model loading
@st.cache_resource
def load_stt_model():
    stt_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-base")
    return stt_pipeline

# Load STT model
stt_pipeline = load_stt_model()

# App title and description
st.title("Audio Emotion Recognition")
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
def predict_emotion(audio_data, sampling_rate):
    # Process audio with transformer processor
    inputs = processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

    # Get model prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=-1).item()

    # Get emotion label and confidence
    emotion = model.config.id2label[predicted_class_id]
    confidence = torch.softmax(logits, dim=1)[0, predicted_class_id].item()

    # Get all emotion probabilities
    probs = torch.softmax(logits, dim=1)[0].tolist()
    all_emotions = {model.config.id2label[i]: prob for i, prob in enumerate(probs)}

    return emotion, confidence, all_emotions

# Function to transcribe audio using Whisper
def transcribe_audio(audio_data_np, sampling_rate, stt_pipeline_obj):
    if audio_data_np is None:
        return "Error: No audio data provided."
    try:
        input_dict = {"raw": audio_data_np, "sampling_rate": sampling_rate}
        result = stt_pipeline_obj(input_dict)
        return result["text"]
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return "Transcription failed."

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

        # Transcribe audio
        transcribed_text = transcribe_audio(audio_data, sr, stt_pipeline)

        # Make prediction
        emotion, confidence, all_emotions = predict_emotion(audio_data, sr)

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
        st.subheader("Audio Waveform")
        waveform_fig = plot_waveform(audio_data, sr)
        st.pyplot(waveform_fig)

    with col2:
        # Display all emotions probabilities
        st.subheader("Emotion Probabilities")
        probs_fig = plot_emotion_probs(all_emotions)
        st.pyplot(probs_fig)

        # Display spectrogram
        st.subheader("Audio Spectrogram")
        spec_fig = plot_spectrogram(audio_data, sr)
        st.pyplot(spec_fig)

    # Display Transcribed Text in a dedicated section
    st.subheader("Transcription")
    st.text_area("Transcription", value=transcribed_text, height=150, disabled=True, help="Transcribed text from the audio.")

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
