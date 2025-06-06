import streamlit as st
import numpy as np
import torch
import librosa
import matplotlib.pyplot as plt
import whisper
from transformers import pipeline
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="Noyce Demo: STT + Emotion Recognition",
    page_icon="🎵",
    layout="wide"
)

# Cache the model loading to improve performance
@st.cache_resource
def load_models():
    # Load both text and audio emotion classifiers
    text_emotion_classifier = None
    audio_emotion_classifier = None
    
    # Try to load text-based emotion classifier
    try:
        text_emotion_classifier = pipeline(
            "text-classification", 
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
    except Exception as e:
        st.warning(f"Could not load text emotion classifier: {e}")
    
    # Load audio-based emotion classifier
    try:
        audio_emotion_classifier = pipeline(
            "audio-classification", 
            model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
            return_all_scores=True
        )
    except Exception as e:
        st.warning(f"Could not load audio emotion classifier: {e}")
        # Try alternative audio model
        try:
            audio_emotion_classifier = pipeline(
                "audio-classification",
                model="superb/wav2vec2-base-superb-er"
            )
        except Exception as e2:
            st.error(f"Could not load any audio emotion classifier: {e2}")
    
    # Load Whisper model for STT
    whisper_model = whisper.load_model("base")
    
    return text_emotion_classifier, audio_emotion_classifier, whisper_model

# Load models
text_emotion_classifier, audio_emotion_classifier, whisper_model = load_models()

# App title and description
st.title("🎵 Speech-to-Text + Emotion Recognition")
st.markdown("Upload an audio file to get both transcribed text and detected emotions from speech.")

# Function to preprocess audio for emotion recognition
def preprocess_audio_for_emotion(audio_file, target_sr=16000):
    y, sr = librosa.load(audio_file, sr=None)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y, target_sr

# Function to transcribe audio using Whisper
def transcribe_audio(audio_file, model):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Transcribe using Whisper
        result = model.transcribe(tmp_path)
        transcribed_text = result["text"]
        language = result["language"]
        
        # Get segments with timestamps if available
        segments = result.get("segments", [])
        
        return transcribed_text, language, segments
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# Function to predict emotion from both text and audio
def predict_emotion_combined(audio_data, sampling_rate, text_classifier, audio_classifier, transcribed_text="", debug=False):
    text_emotion = None
    audio_emotion = None
    text_confidence = 0
    audio_confidence = 0
    
    # Try text-based emotion recognition first
    if text_classifier and transcribed_text.strip():
        try:
            if debug:
                st.write("🔍 **Debug:** Trying text-based emotion recognition")
                st.write(f"🔍 **Debug:** Text input: '{transcribed_text[:100]}...'")
            
            text_predictions = text_classifier(transcribed_text)
            
            if isinstance(text_predictions, list) and len(text_predictions) > 0:
                if isinstance(text_predictions[0], list):
                    # Handle nested list format
                    text_predictions = text_predictions[0]
                
                # Sort by score to get highest confidence
                text_predictions = sorted(text_predictions, key=lambda x: x['score'], reverse=True)
                text_emotion = text_predictions[0]['label']
                text_confidence = text_predictions[0]['score']
                
                if debug:
                    st.write(f"🔍 **Debug:** Text emotion: {text_emotion} ({text_confidence:.3f})")
            
        except Exception as e:
            if debug:
                st.write(f"🔍 **Debug:** Text emotion failed: {str(e)}")
    
    # Try audio-based emotion recognition
    if audio_classifier:
        try:
            if debug:
                st.write("🔍 **Debug:** Trying audio-based emotion recognition")
                st.write(f"🔍 **Debug:** Audio length: {len(audio_data)} samples, {len(audio_data)/sampling_rate:.2f} seconds")
            
            # Preprocess audio for audio model
            max_length = 10 * sampling_rate
            min_length = 1 * sampling_rate
            
            if len(audio_data) > max_length:
                start_idx = (len(audio_data) - max_length) // 2
                audio_data = audio_data[start_idx:start_idx + max_length]
                if debug:
                    st.write("🔍 **Debug:** Audio truncated to 10 seconds")
            elif len(audio_data) < min_length:
                padding = min_length - len(audio_data)
                audio_data = np.pad(audio_data, (0, padding), mode='constant')
                if debug:
                    st.write("🔍 **Debug:** Audio padded to 1 second")
            
            # Normalize audio
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            audio_predictions = audio_classifier({"raw": audio_data, "sampling_rate": sampling_rate})
            
            if isinstance(audio_predictions, list) and len(audio_predictions) > 0:
                if isinstance(audio_predictions[0], list):
                    audio_predictions = audio_predictions[0]
                
                audio_predictions = sorted(audio_predictions, key=lambda x: x['score'], reverse=True)
                audio_emotion = audio_predictions[0]['label']
                audio_confidence = audio_predictions[0]['score']
                
                if debug:
                    st.write(f"🔍 **Debug:** Audio emotion: {audio_emotion} ({audio_confidence:.3f})")
            
        except Exception as e:
            if debug:
                st.write(f"🔍 **Debug:** Audio emotion failed: {str(e)}")
    
    # Decide which result to use
    if text_emotion and audio_emotion:
        # Use the one with higher confidence
        if text_confidence > audio_confidence:
            final_emotion = text_emotion
            final_confidence = text_confidence
            method_used = "text"
        else:
            final_emotion = audio_emotion
            final_confidence = audio_confidence
            method_used = "audio"
    elif text_emotion:
        final_emotion = text_emotion
        final_confidence = text_confidence
        method_used = "text"
    elif audio_emotion:
        final_emotion = audio_emotion
        final_confidence = audio_confidence
        method_used = "audio"
    else:
        # Fallback
        final_emotion = "neutral"
        final_confidence = 0.5
        method_used = "fallback"
    
    # Create combined emotions dict
    all_emotions = {}
    if text_emotion:
        all_emotions[f"{text_emotion}_text"] = text_confidence
    if audio_emotion:
        all_emotions[f"{audio_emotion}_audio"] = audio_confidence
    
    if not all_emotions:
        all_emotions = {"neutral": 0.5}
    
    if debug:
        st.write(f"🔍 **Debug:** Final choice: {final_emotion} from {method_used} ({final_confidence:.3f})")
    
    return final_emotion, final_confidence, all_emotions, method_used

# Function to plot audio waveform
def plot_waveform(audio_data, sr):
    fig, ax = plt.subplots(figsize=(12, 3))
    librosa.display.waveshow(audio_data, sr=sr, ax=ax)
    ax.set_title("Audio Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    return fig

# Function to plot emotion probabilities
def plot_emotion_probs(emotions_dict):
    sorted_emotions = sorted(emotions_dict.items(), key=lambda x: x[1], reverse=True)
    labels, values = zip(*sorted_emotions)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color='lightblue')
    bars[0].set_color('darkblue')  # Highlight highest probability
    
    ax.set_title("Emotion Probabilities")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.0)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# Sidebar configuration
st.sidebar.title("⚙️ Settings")

# Debug mode toggle
debug_mode = st.sidebar.checkbox("🐛 Debug Mode", value=False, help="Show detailed processing information")

# Model selection
whisper_model_size = st.sidebar.selectbox(
    "Whisper Model Size",
    ["tiny", "base", "small", "medium", "large"],
    index=1,
    help="Larger models are more accurate but slower"
)

# Language selection (optional)
detect_language = st.sidebar.checkbox("Auto-detect language", value=True)

if not detect_language:
    language_code = st.sidebar.selectbox(
        "Language",
        ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "zh"],
        help="Select the expected language of the audio"
    )

# Main app functionality
uploaded_file = st.file_uploader(
    "Choose an audio file", 
    type=["wav", "mp3", "ogg", "m4a", "flac"]
)

if uploaded_file is not None:
    # Display audio player
    st.audio(uploaded_file, format="audio/wav")
    
    # Process audio
    with st.spinner("🔄 Processing audio (STT + Emotion Analysis)..."):
        # Speech-to-Text using Whisper
        transcribed_text, detected_language, segments = transcribe_audio(
            uploaded_file, whisper_model
        )
        
        # Reset file pointer for emotion analysis
        uploaded_file.seek(0)
        
        # Preprocess for emotion recognition
        audio_data, sr = preprocess_audio_for_emotion(uploaded_file)
        
        # Emotion prediction using both text and audio
        emotion, confidence, all_emotions, method_used = predict_emotion_combined(
            audio_data, sr, text_emotion_classifier, audio_emotion_classifier, transcribed_text, debug_mode
        )
    
    # Display results
    st.success("✅ Analysis complete!")
    st.write(f"**Emotion detection method:** {method_used}-based")
    
    # Main results section
    st.markdown("## 📝 Transcription + 😊 Emotion Results")
    
    # Create columns for main results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Transcribed text
        st.subheader("📝 Transcribed Text")
        st.markdown(f"**Language detected:** {detected_language}")
        
        # Display transcribed text in a nice box
        st.success("**Transcription:**")
        st.markdown(f"**{transcribed_text}**")
    
    with col2:
        # Detected emotion
        st.subheader("😊 Detected Emotion")
        
        # Map emotions to emojis
        emoji_map = {
            "angry": "😠", "disgust": "🤢", "fear": "😨", 
            "happy": "😃", "neutral": "😐", "sad": "😢", 
            "surprise": "😲", "calm": "😌", "excited": "🤩",
            "frustrated": "😤", "other": "❓"
        }
        
        emoji = emoji_map.get(emotion.lower(), "❓")
        
        st.markdown(f"## {emoji}")
        st.markdown(f"### {emotion.capitalize()}")
        st.markdown(f"**Confidence:** {confidence:.2f}")
    
    # Combined output section
    st.markdown("## 🎯 Combined Output")
    
    # Use Streamlit's info box for better visibility
    st.info(f"""
    **📋 Complete Analysis:**
    
    **🗣️ Transcribed Text:** {transcribed_text}
    
    **😊 Detected Emotion:** {emotion.capitalize()} (Confidence: {confidence:.2f})
    
    **🌍 Language:** {detected_language}
    """)
    
    # Detailed analysis section
    with st.expander("🔍 Detailed Analysis", expanded=False):
        col3, col4 = st.columns(2)
        
        with col3:
            # Audio waveform
            st.subheader("🌊 Waveform")
            waveform_fig = plot_waveform(audio_data, sr)
            st.pyplot(waveform_fig)
            
            # Audio details
            st.subheader("ℹ️ Audio Details")
            st.write(f"**Duration:** {len(audio_data)/sr:.2f} seconds")
            st.write(f"**Sampling Rate:** {sr} Hz")
            st.write(f"**Total Samples:** {len(audio_data):,}")
        
        with col4:
            # Emotion probabilities
            st.subheader("📊 All Emotion Probabilities")
            probs_fig = plot_emotion_probs(all_emotions)
            st.pyplot(probs_fig)
            
            # Show segments if available
            if segments:
                st.subheader("⏱️ Transcription Segments")
                for i, segment in enumerate(segments[:5]):  # Show first 5 segments
                    start_time = segment.get('start', 0)
                    end_time = segment.get('end', 0)
                    text = segment.get('text', '')
                    st.write(f"**{start_time:.1f}s - {end_time:.1f}s:** {text}")

# Information sections
st.markdown("---")

# Usage examples
st.subheader("💡 Usage Examples")

tab1, tab2, tab3 = st.tabs(["📝 Business Meeting", "📞 Customer Service", "🎓 Education"])

with tab1:
    st.markdown("""
    **Business Meeting Analysis:**
    - **Input:** Recording of a team meeting
    - **Output:** 
      - Text: "I think we should proceed with the new marketing strategy..."
      - Emotion: Confident (0.84)
    - **Use Case:** Meeting transcription with sentiment analysis
    """)

with tab2:
    st.markdown("""
    **Customer Service Call:**
    - **Input:** Customer support conversation
    - **Output:**
      - Text: "I'm really frustrated with this product..."
      - Emotion: Angry (0.91)
    - **Use Case:** Customer satisfaction monitoring
    """)

with tab3:
    st.markdown("""
    **Educational Content:**
    - **Input:** Student presentation or lecture
    - **Output:**
      - Text: "Today we'll learn about machine learning..."
      - Emotion: Neutral (0.78)
    - **Use Case:** Educational content analysis
    """)

# Sidebar information
st.sidebar.markdown("---")
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    "This app combines OpenAI Whisper for speech-to-text with Hugging Face transformers "
    "for emotion recognition. Upload audio files to get both transcription and emotion analysis."
)

st.sidebar.title("🚀 Performance Tips")
st.sidebar.markdown("""
**For better results:**
- Use clear audio with minimal background noise
- Audio should be primarily speech
- Shorter clips (30 seconds - 5 minutes) process faster
- The 'base' Whisper model balances speed and accuracy well

**Model sizes:**
- `tiny`: Fastest, least accurate
- `base`: Good balance (recommended)
- `large`: Most accurate, slowest
""")

st.sidebar.title("📁 Sample Data")
st.sidebar.markdown("""
**Audio datasets for testing:**
- [Common Voice](https://commonvoice.mozilla.org/)
- [LibriSpeech](http://www.openslr.org/12/)
- [RAVDESS](https://zenodo.org/record/1188976) (with emotions)
""")

# Installation instructions
st.sidebar.title("🔧 Troubleshooting")
st.sidebar.markdown("""
**If emotion detection seems stuck:**
1. Enable Debug Mode to see processing details
2. Check that both text and audio classifiers loaded
3. Try shorter audio clips (2-5 seconds)  
4. Ensure audio has clear speech
5. Check audio volume (not too quiet/loud)
6. Try WAV format for best compatibility

**Recent fix:** Separated text and audio emotion models to prevent file path errors.
""")

st.sidebar.title("🔧 Installation")
with st.sidebar.expander("Required packages"):
    st.code("""
pip install streamlit
pip install openai-whisper
pip install transformers
pip install torch
pip install librosa
pip install matplotlib
    """)