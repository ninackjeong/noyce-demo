import streamlit as st
import numpy as np
import torch
import librosa
import whisper
from transformers import pipeline
import tempfile
import os
import requests
from gtts import gTTS
import io
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Audio → Emotion + STT → LLM → TTS",
    page_icon="🎤",
    layout="wide"
)

# Cache the model loading to prevent reloading
@st.cache_resource
def load_models():
    """Load AI models for speech recognition and emotion detection"""
    
    # Load Whisper model for speech-to-text
    whisper_model = whisper.load_model("base")
    
    # Load pretrained audio emotion classifier
    audio_emotion_classifier = None
    try:
        audio_emotion_classifier = pipeline(
            "audio-classification", 
            model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
            return_all_scores=True
        )
        st.success("✅ Audio emotion model loaded successfully")
    except Exception as e:
        st.error(f"❌ Could not load audio emotion model: {e}")
    
    return whisper_model, audio_emotion_classifier

# Load models once at startup
whisper_model, audio_emotion_classifier = load_models()

# App header
st.title("🎤 Noyce Demo: Empathetic Response Generation with Emotion Recognition")
st.markdown("**Audio → STT + Emotion Detection → LLM Response → TTS**")

# Sidebar configuration
st.sidebar.title("⚙️ Configuration")

# OpenAI API configuration
st.sidebar.subheader("🤖 LLM Settings")
use_openai = st.sidebar.checkbox("🔥 Use OpenAI GPT (recommended)", value=True)

if use_openai:
    openai_key = st.sidebar.text_input(
        "OpenAI API Key", 
        type="password",
        help="Get your API key from https://platform.openai.com/api-keys",
        placeholder="sk-..."
    )
    
    if openai_key:
        # Test API key validity
        try:
            headers = {"Authorization": f"Bearer {openai_key}"}
            test_response = requests.get(
                "https://api.openai.com/v1/models", 
                headers=headers, 
                timeout=5
            )
            if test_response.status_code == 200:
                st.sidebar.success("✅ API key is valid")
            else:
                st.sidebar.error("❌ Invalid API key")
        except:
            st.sidebar.warning("⚠️ Could not verify API key")
    else:
        st.sidebar.info("💡 Add your OpenAI API key for best results")
else:
    st.sidebar.info("Using simple fallback responses")

# Response style configuration
response_style = st.sidebar.selectbox(
    "Response Style",
    ["Empathetic", "Casual", "Professional", "Supportive"],
    help="How should the AI respond to detected emotions?"
)

# TTS configuration
use_tts = st.sidebar.checkbox("🔊 Generate Speech Response", value=True)

# Manual emotion override option
st.sidebar.subheader("🎭 Emotion Settings")
enable_manual_emotion = st.sidebar.checkbox(
    "✋ Manual Emotion Override", 
    value=False,
    help="Override automatic emotion detection with manual selection"
)

if enable_manual_emotion:
    manual_emotion = st.sidebar.selectbox(
        "Select Emotion",
        ["angry", "sad", "happy", "fear", "neutral", "surprise", "disgust"],
        index=4  # default to neutral
    )

# Debug mode
debug = st.sidebar.checkbox("🐛 Debug Mode", help="Show detailed processing information")

# Core functions
def transcribe_audio(audio_file):
    """
    Convert audio to text using Whisper STT model
    
    Args:
        audio_file: Uploaded audio file
        
    Returns:
        str: Transcribed text
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        if debug:
            st.write(f"🔍 Transcribing audio file: {tmp_path}")
        
        result = whisper_model.transcribe(tmp_path)
        transcribed_text = result["text"].strip()
        
        if debug:
            st.write(f"🔍 Transcription result: {transcribed_text}")
        
        return transcribed_text
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        return ""
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def get_emotion_from_audio(audio_file):
    """
    Detect emotion from audio using pretrained model
    
    Args:
        audio_file: Uploaded audio file
        
    Returns:
        tuple: (emotion_label, confidence_score)
    """
    if not audio_emotion_classifier:
        if debug:
            st.write("🔍 No emotion classifier available, returning neutral")
        return "neutral", 0.5
    
    try:
        if debug:
            st.write("🔍 Starting emotion detection from audio...")
        
        # Load and preprocess audio for emotion detection
        y, sr = librosa.load(audio_file, sr=16000)
        
        # Ensure proper audio length (1-10 seconds for best results)
        if len(y) > 10 * sr:
            y = y[:10 * sr]  # Truncate to 10 seconds
            if debug:
                st.write("🔍 Audio truncated to 10 seconds")
        elif len(y) < sr:
            y = np.pad(y, (0, sr - len(y)))  # Pad to 1 second minimum
            if debug:
                st.write("🔍 Audio padded to 1 second")
        
        # Normalize audio amplitude
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        
        # Get emotion predictions from the model
        predictions = audio_emotion_classifier({"raw": y, "sampling_rate": sr})
        
        if predictions and len(predictions) > 0:
            # Sort predictions by confidence and get the top emotion
            sorted_predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
            top_emotion = sorted_predictions[0]
            
            if debug:
                st.write(f"🔍 Top 3 emotion predictions:")
                for i, pred in enumerate(sorted_predictions[:3]):
                    st.write(f"   {i+1}. {pred['label']}: {pred['score']:.3f}")
            
            return top_emotion['label'], top_emotion['score']
        
    except Exception as e:
        if debug:
            st.write(f"🔍 Emotion detection failed: {e}")
        st.warning(f"Emotion detection failed: {e}")
    
    return "neutral", 0.5

def generate_llm_response(text, emotion, confidence, style):
    """
    Generate contextual response using LLM based on text and emotion
    
    Args:
        text (str): User's transcribed speech
        emotion (str): Detected or manually selected emotion
        confidence (float): Confidence score of emotion detection
        style (str): Response style preference
        
    Returns:
        str: Generated response text
    """
    
    # Try OpenAI GPT first if API key is available
    if use_openai and openai_key:
        try:
            if debug:
                st.write("🔍 Generating response using OpenAI GPT...")
            
            headers = {
                "Authorization": f"Bearer {openai_key}",
                "Content-Type": "application/json"
            }
            
            # Craft a natural prompt for the LLM
            prompt = f"""You are a helpful, empathetic and speech aware AI assistant. You are designed to interact with users whose speech may be difficult to understand. 
            Their words might:
            - Be unclear or fragmented
            - Not make logical sense or follow a clear topic
            - Include repetitive stories or phrases
            - Be monotonous, hesitant, or slow
            - Reflect emotional distress, fatigue, or cognitive challenges

            The user said: "{text}"

            Their detected emotion is: {emotion} (confidence: {confidence:.1f})
            Response style: {style}

            Generate a natural, helpful response that acknowledges their emotional state.
            DO NOT quote or repeat their exact words back to them.
            Just respond naturally and appropriately based on what they said and how they're feeling.
            Keep it 2-3 sentences maximum.
            """

            # API request payload
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are a {style.lower()}, empathetic AI assistant."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "max_tokens": 120,
                "temperature": 0.7,
                "presence_penalty": 0.2,
                "frequency_penalty": 0.2
            }
            
            # Make API request to OpenAI
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers, 
                json=data, 
                timeout=15
            )
            
            if debug:
                st.write(f"🔍 OpenAI API status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result["choices"][0]["message"]["content"].strip()
                
                if debug:
                    st.write(f"🔍 OpenAI response: {generated_text}")
                
                # Clean text for TTS compatibility
                return clean_text_for_tts(generated_text)
            else:
                error_detail = response.json().get('error', {}).get('message', 'Unknown error')
                st.error(f"OpenAI API Error ({response.status_code}): {error_detail}")
                
        except requests.exceptions.Timeout:
            st.error("OpenAI API request timed out")
        except Exception as e:
            if debug:
                st.write(f"🔍 OpenAI API failed: {e}")
            st.error(f"OpenAI API failed: {e}")
    
    # Fallback to simple response generation
    if debug:
        st.write("🔍 Using fallback response generation...")
    return generate_simple_fallback(emotion, style)

def generate_simple_fallback(emotion, style):
    """
    Generate simple fallback responses when LLM is unavailable
    
    Args:
        emotion (str): Detected emotion
        style (str): Response style
        
    Returns:
        str: Simple contextual response
    """
    
    # Emotion-based response templates
    responses = {
        "angry": {
            "empathetic": "I can sense your frustration. That must be really difficult to deal with.",
            "casual": "That sounds super frustrating! I totally get why you'd be upset.",
            "professional": "I understand your concerns. How can I help address this issue?",
            "supportive": "I hear your frustration, and those feelings are completely valid."
        },
        "sad": {
            "empathetic": "I'm sorry you're going through this. That sounds really tough.",
            "casual": "Aw, that's really hard. I'm here if you need to talk about it.",
            "professional": "I understand this is difficult. What support would be most helpful?",
            "supportive": "It takes courage to share when you're feeling down. I'm here for you."
        },
        "happy": {
            "empathetic": "That's wonderful! I can feel your joy and excitement.",
            "casual": "That's awesome! I'm so happy for you!",
            "professional": "Congratulations! That's excellent news.",
            "supportive": "Your happiness is contagious! That's fantastic to hear."
        },
        "fear": {
            "empathetic": "I understand why that feels worrying. Those concerns are completely valid.",
            "casual": "That does sound nerve-wracking, but I believe you can handle it!",
            "professional": "I recognize your concerns. Let's work through this systematically.",
            "supportive": "Feeling anxious about this is normal. You're not alone in this."
        },
        "surprise": {
            "empathetic": "That must have been quite unexpected! How are you processing this?",
            "casual": "Whoa, that's surprising! What a twist!",
            "professional": "That's an unexpected development. How can I help you navigate this?",
            "supportive": "Surprises can be overwhelming. Take your time to process this."
        },
        "disgust": {
            "empathetic": "I can understand why that would be off-putting. That's a reasonable reaction.",
            "casual": "Ugh, that sounds pretty gross or annoying!",
            "professional": "I understand your dissatisfaction with this situation.",
            "supportive": "Your reaction is completely understandable. That does sound unpleasant."
        },
        "neutral": {
            "empathetic": "Thanks for sharing that. How are you feeling about the situation?",
            "casual": "Got it! What's your take on that?",
            "professional": "I understand. How can I best assist you with this?",
            "supportive": "I appreciate you sharing. What would be most helpful right now?"
        }
    }
    
    # Get appropriate response based on emotion and style
    emotion_responses = responses.get(emotion.lower(), responses["neutral"])
    return emotion_responses.get(style.lower(), emotion_responses["empathetic"])

def clean_text_for_tts(text):
    """
    Clean text for better TTS pronunciation by removing problematic characters
    
    Args:
        text (str): Raw text from LLM
        
    Returns:
        str: Cleaned text suitable for TTS
    """
    import re
    
    # Remove quotes and problematic punctuation for TTS
    text = text.replace('"', '').replace("'", '')
    
    # Remove content in parentheses (like confidence scores)
    text = re.sub(r'\([^)]*\)', '', text)
    
    # Clean up multiple spaces and normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def text_to_speech(text):
    """
    Convert text to speech using Google TTS
    
    Args:
        text (str): Text to convert to speech
        
    Returns:
        bytes: MP3 audio data, or None if failed
    """
    try:
        if debug:
            st.write(f"🔍 Converting to speech: {text}")
        
        # Create TTS object and generate speech
        tts = gTTS(text=text, lang="en", slow=False)
        
        # Save to bytes buffer
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        
        if debug:
            st.write("🔍 TTS generation successful")
        
        return mp3_fp.getvalue()
    except Exception as e:
        if debug:
            st.write(f"🔍 TTS failed: {e}")
        st.error(f"Text-to-speech failed: {e}")
        return None

# Quick test section for development and debugging
st.markdown("---")
st.subheader("🧪 Quick Response Test")
st.markdown("Test the LLM response generation without uploading audio:")

col1, col2, col3 = st.columns(3)

with col1:
    test_text = st.text_input(
        "Test text:", 
        "I'm really stressed about work",
        help="Enter sample text to test response generation"
    )

with col2:
    test_emotion = st.selectbox(
        "Test emotion:", 
        ["angry", "sad", "happy", "fear", "neutral", "surprise", "disgust"],
        index=0,
        help="Select emotion to test with"
    )

with col3:
    if st.button("🎯 Test Response Generation"):
        with st.spinner("Generating test response..."):
            test_response = generate_llm_response(test_text, test_emotion, 0.8, response_style)
        st.success(f"**Generated Response:** {test_response}")

# Main application interface
st.markdown("---")
st.subheader("🎤 Full Audio Processing Pipeline")

uploaded_file = st.file_uploader(
    "Upload your audio file", 
    type=["wav", "mp3", "ogg", "m4a", "flac"],
    help="Record yourself speaking naturally - the AI will detect your emotion and respond appropriately!"
)

if uploaded_file is not None:
    # Display audio player for user reference
    st.audio(uploaded_file, format="audio/wav")
    
    with st.spinner("🔄 Processing your audio through the AI pipeline..."):
        
        # Step 1: Speech-to-Text conversion
        if debug:
            st.write("### 🎯 Step 1: Speech-to-Text Conversion")
        
        transcribed_text = transcribe_audio(uploaded_file)
        
        if not transcribed_text:
            st.error("Failed to transcribe audio. Please try again with a clearer recording.")
            st.stop()
        
        # Step 2: Emotion detection (automatic or manual)
        if debug:
            st.write("### 🎯 Step 2: Emotion Detection")
        
        if enable_manual_emotion:
            # Use manually selected emotion
            emotion = manual_emotion
            confidence = 1.0  # Manual selection has 100% confidence
            if debug:
                st.write(f"🔍 Using manual emotion: {emotion} (confidence: {confidence})")
        else:
            # Use automatic emotion detection
            uploaded_file.seek(0)  # Reset file pointer for emotion detection
            emotion, confidence = get_emotion_from_audio(uploaded_file)
            if debug:
                st.write(f"🔍 Detected emotion: {emotion} (confidence: {confidence:.3f})")
        
        # Step 3: LLM response generation
        if debug:
            st.write("### 🎯 Step 3: LLM Response Generation")
        
        ai_response = generate_llm_response(transcribed_text, emotion, confidence, response_style)
        
        # Step 4: Text-to-Speech conversion (optional)
        speech_audio = None
        if use_tts:
            if debug:
                st.write("### 🎯 Step 4: Text-to-Speech Conversion")
            speech_audio = text_to_speech(ai_response)
    
    # Display results in a clean layout
    st.success("✅ AI Pipeline Processing Complete!")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🗣️ Your Input Analysis")
        
        emotion_source = "Manual Selection" if enable_manual_emotion else "Auto-Detected"
        
        st.info(f"""
        **🎤 You said:** {transcribed_text}
        
        **😊 Emotion:** {emotion.capitalize()} ({confidence:.2f} confidence)
        
        **🎯 Source:** {emotion_source}
        
        **📊 Processing:** Whisper STT + Emotion AI
        """)
    
    with col2:
        st.subheader("🤖 AI Assistant Response")
        
        llm_source = "OpenAI GPT-3.5" if (use_openai and openai_key) else "Simple Fallback"
        
        st.success(f"""
        **💬 AI Response:**
        
        {ai_response}
        
        **🎭 Style:** {response_style}
        
        **🧠 Generated by:** {llm_source}
        """)
    
    # Audio response section
    if speech_audio:
        st.subheader("🔊 AI Voice Response")
        st.audio(speech_audio, format="audio/mp3")
        
        # Download button for the generated audio
        st.download_button(
            label="⬇️ Download AI Response Audio",
            data=speech_audio,
            file_name=f"ai_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3",
            mime="audio/mp3"
        )
    elif use_tts:
        st.warning("⚠️ Text-to-speech generation failed. Response is available as text above.")

# Example results showcase
st.markdown("---")
st.subheader("💡 Example Conversation Results")

examples = [
    {
        "emotion": "😢 Sad", 
        "input": "I lost my job today", 
        "response": "I'm sorry you're going through this. That sounds really tough and challenging."
    },
    {
        "emotion": "😡 Angry", 
        "input": "This software keeps crashing!", 
        "response": "That sounds super frustrating! Technical issues like that are the worst."
    },
    {
        "emotion": "😄 Happy", 
        "input": "I just got promoted!", 
        "response": "That's awesome! Congratulations on your promotion - you must be thrilled!"
    },
    {
        "emotion": "😐 Neutral", 
        "input": "It costs $7 and takes one day", 
        "response": "That sounds reasonable. Do you have any other questions about the process?"
    }
]

for example in examples:
    with st.expander(f"{example['emotion']} - Example Response"):
        st.markdown(f"**🎤 User Input:** {example['input']}")
        st.markdown(f"**🤖 AI Response:** {example['response']}")

# Sidebar information and instructions
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ How It Works")
st.sidebar.info("""
**AI Pipeline Steps:**

1. **🎤 Audio → STT** (Whisper)
2. **🎭 Audio → Emotion** (Pretrained Model)  
3. **🤖 Text + Emotion → LLM** (GPT/Fallback)
4. **🔊 Response → TTS** (Google TTS)

**✨ Key Features:**
- Manual emotion override option
- No awkward text quoting in responses
- Clean audio without special characters
- Natural conversational flow
- OpenAI GPT integration with fallback

Perfect for voice assistants, therapy bots, and customer service!
""")

st.sidebar.subheader("📦 Installation Requirements")
st.sidebar.code("""
pip install streamlit
pip install openai-whisper
pip install transformers
pip install torch
pip install librosa
pip install gtts
pip install requests
""")

st.sidebar.subheader("🔑 API Setup")
st.sidebar.markdown("""
**OpenAI API Key:**
1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create a new API key
3. Add billing information
4. Paste the key above

**Cost:** ~$0.002 per request (very affordable!)
""")