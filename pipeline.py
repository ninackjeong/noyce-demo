import streamlit as st
import numpy as np
import torch
import librosa
import matplotlib.pyplot as plt
import whisper
from transformers import pipeline
import tempfile
import os
import requests
import json
from gtts import gTTS
import io
import base64
from datetime import datetime

# For local LLM (alternative to OpenAI)
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    LOCAL_LLM_AVAILABLE = True
except ImportError:
    LOCAL_LLM_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Complete AI Assistant: STT → Emotion → LLM → TTS",
    page_icon="🤖",
    layout="wide"
)

# Cache the model loading to improve performance
@st.cache_resource
def load_all_models():
    # Load emotion classifiers
    text_emotion_classifier = None
    audio_emotion_classifier = None
    
    # Text emotion classifier
    try:
        text_emotion_classifier = pipeline(
            "text-classification", 
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
    except Exception as e:
        st.warning(f"Could not load text emotion classifier: {e}")
    
    # Audio emotion classifier
    try:
        audio_emotion_classifier = pipeline(
            "audio-classification", 
            model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
            return_all_scores=True
        )
    except Exception as e:
        st.warning(f"Could not load audio emotion classifier: {e}")
    
    # Whisper for STT
    whisper_model = whisper.load_model("base")
    
    # Local LLM for response generation (optional)
    local_llm_tokenizer = None
    local_llm_model = None
    
    if LOCAL_LLM_AVAILABLE:
        try:
            # Using a smaller, faster model for response generation
            model_name = "microsoft/DialoGPT-medium"
            local_llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            local_llm_model = AutoModelForCausalLM.from_pretrained(model_name)
            local_llm_tokenizer.pad_token = local_llm_tokenizer.eos_token
        except Exception as e:
            st.warning(f"Could not load local LLM: {e}")
    
    return (text_emotion_classifier, audio_emotion_classifier, whisper_model, 
            local_llm_tokenizer, local_llm_model)

# Load all models
(text_emotion_classifier, audio_emotion_classifier, whisper_model, 
 local_llm_tokenizer, local_llm_model) = load_all_models()

# App title and description
st.title("🤖 Complete AI Assistant Pipeline")
st.markdown("**STT → Emotion Recognition → LLM Response → TTS**")
st.markdown("Upload audio to get an intelligent, emotion-aware response in both text and speech!")

# Sidebar configuration
st.sidebar.title("⚙️ Configuration")

# LLM Selection
llm_choice = st.sidebar.selectbox(
    "Choose LLM for Response Generation",
    ["OpenAI GPT", "Local DialoGPT", "Hugging Face API", "Custom Prompt Only"],
    help="OpenAI requires API key, Local is free but less sophisticated"
)

# API Key input if needed
openai_api_key = None
hf_api_key = None

if llm_choice == "OpenAI GPT":
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key", 
        type="password",
        help="Get your API key from https://platform.openai.com/"
    )

elif llm_choice == "Hugging Face API":
    hf_api_key = st.sidebar.text_input(
        "Hugging Face API Key",
        type="password", 
        help="Get your API key from https://huggingface.co/settings/tokens"
    )

# TTS Options
tts_choice = st.sidebar.selectbox(
    "Choose TTS Engine",
    ["Google TTS (gTTS)", "OpenAI TTS", "Local pyttsx3"],
    help="gTTS is free and good quality, OpenAI TTS requires API key"
)

# Voice selection for different TTS engines
if tts_choice == "OpenAI TTS":
    voice_choice = st.sidebar.selectbox(
        "Voice",
        ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    )
elif tts_choice == "Google TTS (gTTS)":
    voice_choice = st.sidebar.selectbox(
        "Language/Accent",
        ["en", "en-us", "en-uk", "en-au", "en-ca"]
    )

# Response style
response_style = st.sidebar.selectbox(
    "Response Style",
    ["Empathetic", "Professional", "Casual", "Therapeutic", "Educational"],
    help="How should the AI respond based on detected emotion?"
)

debug_mode = st.sidebar.checkbox("🐛 Debug Mode", value=False)

# Function to transcribe audio using Whisper
def transcribe_audio(audio_file, model):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        result = model.transcribe(tmp_path)
        return result["text"], result["language"], result.get("segments", [])
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# Function to predict emotion (same as before)
def predict_emotion_combined(audio_data, sampling_rate, text_classifier, audio_classifier, transcribed_text="", debug=False):
    text_emotion = None
    audio_emotion = None
    text_confidence = 0
    audio_confidence = 0
    
    # Try text-based emotion recognition
    if text_classifier and transcribed_text.strip():
        try:
            if debug:
                st.write("🔍 **Debug:** Trying text-based emotion recognition")
            
            text_predictions = text_classifier(transcribed_text)
            
            if isinstance(text_predictions, list) and len(text_predictions) > 0:
                if isinstance(text_predictions[0], list):
                    text_predictions = text_predictions[0]
                
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
            
            # Preprocess audio
            max_length = 10 * sampling_rate
            min_length = 1 * sampling_rate
            
            if len(audio_data) > max_length:
                start_idx = (len(audio_data) - max_length) // 2
                audio_data = audio_data[start_idx:start_idx + max_length]
            elif len(audio_data) < min_length:
                padding = min_length - len(audio_data)
                audio_data = np.pad(audio_data, (0, padding), mode='constant')
            
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
    
    # Choose best result
    if text_emotion and audio_emotion:
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
        final_emotion = "neutral"
        final_confidence = 0.5
        method_used = "fallback"
    
    if debug:
        st.write(f"🔍 **Debug:** Final emotion: {final_emotion} from {method_used} ({final_confidence:.3f})")
    
    return final_emotion, final_confidence, method_used

# Function to generate emotion-aware response
def generate_response(text, emotion, confidence, style, llm_choice, api_key=None, debug=False):
    
    # Create emotion-aware prompt
    emotion_context = {
        "angry": "The user sounds frustrated or angry",
        "sad": "The user sounds sad or disappointed", 
        "happy": "The user sounds happy and positive",
        "fear": "The user sounds worried or anxious",
        "surprise": "The user sounds surprised",
        "disgust": "The user sounds disgusted or annoyed",
        "neutral": "The user has a neutral emotional tone",
        "calm": "The user sounds calm and composed",
        "excited": "The user sounds excited and energetic"
    }
    
    style_instructions = {
        "Empathetic": "Respond with empathy and understanding. Acknowledge their emotions.",
        "Professional": "Respond in a professional, helpful manner.",
        "Casual": "Respond in a friendly, casual tone.",
        "Therapeutic": "Respond like a supportive counselor, offering gentle guidance.",
        "Educational": "Respond with informative, educational content."
    }
    
    emotion_desc = emotion_context.get(emotion.lower(), f"The user sounds {emotion}")
    style_inst = style_instructions.get(style, "Respond appropriately")
    
    system_prompt = f"""
You are an AI assistant that responds appropriately to human emotions. 

Context: {emotion_desc} (confidence: {confidence:.2f}).
Instructions: {style_inst}
User said: "{text}"

Provide a helpful, appropriate response (2-3 sentences max).
"""

    if debug:
        st.write(f"🔍 **Debug:** Generated prompt: {system_prompt[:200]}...")
    
    # Generate response based on chosen LLM
    try:
        if llm_choice == "OpenAI GPT" and api_key:
            response = generate_openai_response(system_prompt, api_key, debug)
        elif llm_choice == "Local DialoGPT" and local_llm_model:
            response = generate_local_response(text, emotion, local_llm_tokenizer, local_llm_model, debug)
        elif llm_choice == "Hugging Face API" and api_key:
            response = generate_huggingface_response(system_prompt, api_key, debug)
        else:
            # Fallback: rule-based response
            response = generate_rule_based_response(text, emotion, style, debug)
        
        return response
        
    except Exception as e:
        if debug:
            st.write(f"🔍 **Debug:** LLM generation failed: {str(e)}")
        return generate_rule_based_response(text, emotion, style, debug)

def generate_openai_response(prompt, api_key, debug=False):
    """Generate response using OpenAI API"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0.7
    }
    
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=data
    )
    
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    else:
        raise Exception(f"OpenAI API error: {response.status_code}")

def generate_local_response(text, emotion, tokenizer, model, debug=False):
    """Generate response using local DialoGPT"""
    # Simple prompt for DialoGPT
    input_text = f"User ({emotion}): {text} Bot:"
    
    # Encode and generate
    inputs = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + 50,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )
    
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return response.strip()

def generate_huggingface_response(prompt, api_key, debug=False):
    """Generate response using Hugging Face Inference API"""
    headers = {"Authorization": f"Bearer {api_key}"}
    
    # Using a good conversational model
    API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-large"
    
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "").strip()
        return str(result)
    else:
        raise Exception(f"Hugging Face API error: {response.status_code}")

def generate_rule_based_response(text, emotion, style, debug=False):
    """Fallback rule-based response generator"""
    
    if debug:
        st.write("🔍 **Debug:** Using rule-based response generation")
    
    # Emotion-specific responses
    responses = {
        "angry": [
            "I understand you're feeling frustrated. Let's work through this together.",
            "I can sense your frustration. How can I help address your concerns?",
            "It sounds like this is really bothering you. What would help most right now?"
        ],
        "sad": [
            "I'm sorry you're feeling down. Is there anything I can do to help?",
            "That sounds difficult. Thank you for sharing this with me.",
            "I hear that you're going through a tough time. I'm here to listen."
        ],
        "happy": [
            "That's wonderful to hear! I'm glad you're feeling positive.",
            "It's great to hear the enthusiasm in your voice!",
            "That sounds fantastic! Tell me more about what's making you happy."
        ],
        "fear": [
            "I understand this feels worrying. Let's break it down step by step.",
            "It's natural to feel anxious about this. What specific concerns do you have?",
            "I can hear the concern in your voice. How can I help ease your worries?"
        ],
        "neutral": [
            "Thank you for sharing that. How can I best assist you today?",
            "I understand. What would you like to know more about?",
            "That's an interesting point. What are your thoughts on next steps?"
        ]
    }
    
    # Get appropriate responses for emotion
    emotion_responses = responses.get(emotion.lower(), responses["neutral"])
    
    # Pick response based on style
    if style == "Professional":
        return emotion_responses[1] if len(emotion_responses) > 1 else emotion_responses[0]
    elif style == "Casual":
        return emotion_responses[0]
    else:
        return emotion_responses[-1]

# Function to convert text to speech
def text_to_speech(text, tts_choice, voice_choice=None, api_key=None, debug=False):
    """Convert text to speech using various TTS engines"""
    
    if debug:
        st.write(f"🔍 **Debug:** Converting to speech using {tts_choice}")
    
    try:
        if tts_choice == "Google TTS (gTTS)":
            # Use Google Text-to-Speech
            tts = gTTS(text=text, lang=voice_choice or "en", slow=False)
            
            # Save to bytes
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            
            return mp3_fp.getvalue(), "audio/mp3"
            
        elif tts_choice == "OpenAI TTS" and api_key:
            # Use OpenAI TTS API
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "tts-1",
                "input": text,
                "voice": voice_choice or "alloy"
            }
            
            response = requests.post(
                "https://api.openai.com/v1/audio/speech",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                return response.content, "audio/mp3"
            else:
                raise Exception(f"OpenAI TTS error: {response.status_code}")
                
        else:
            # Fallback to simple text display
            if debug:
                st.write("🔍 **Debug:** TTS not available, returning text only")
            return None, None
            
    except Exception as e:
        if debug:
            st.write(f"🔍 **Debug:** TTS failed: {str(e)}")
        return None, None

# Function to preprocess audio for emotion recognition
def preprocess_audio_for_emotion(audio_file, target_sr=16000):
    y, sr = librosa.load(audio_file, sr=None)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y, target_sr

# Main app functionality
uploaded_file = st.file_uploader(
    "🎤 Upload your audio message", 
    type=["wav", "mp3", "ogg", "m4a", "flac"],
    help="Speak naturally - the AI will detect your emotion and respond appropriately!"
)

if uploaded_file is not None:
    # Display audio player
    st.audio(uploaded_file, format="audio/wav")
    
    # Process the complete pipeline
    with st.spinner("🔄 Processing complete AI pipeline..."):
        
        # Step 1: Speech-to-Text
        if debug_mode:
            st.write("### 🎯 Step 1: Speech-to-Text")
        transcribed_text, detected_language, segments = transcribe_audio(uploaded_file, whisper_model)
        
        # Step 2: Emotion Recognition
        if debug_mode:
            st.write("### 🎯 Step 2: Emotion Recognition")
        uploaded_file.seek(0)  # Reset file pointer
        audio_data, sr = preprocess_audio_for_emotion(uploaded_file)
        emotion, confidence, method_used = predict_emotion_combined(
            audio_data, sr, text_emotion_classifier, audio_emotion_classifier, 
            transcribed_text, debug_mode
        )
        
        # Step 3: Generate AI Response
        if debug_mode:
            st.write("### 🎯 Step 3: Generate AI Response")
        ai_response = generate_response(
            transcribed_text, emotion, confidence, response_style, 
            llm_choice, openai_api_key or hf_api_key, debug_mode
        )
        
        # Step 4: Text-to-Speech
        if debug_mode:
            st.write("### 🎯 Step 4: Text-to-Speech")
        speech_audio, audio_format = text_to_speech(
            ai_response, tts_choice, voice_choice, 
            openai_api_key, debug_mode
        )
    
    # Display results
    st.success("✅ Complete AI Pipeline Processed!")
    
    # Create main result display
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🗣️ Your Input")
        st.info(f"""
        **🎤 You said:** {transcribed_text}
        
        **😊 Detected emotion:** {emotion.capitalize()} ({confidence:.2f})
        
        **🌍 Language:** {detected_language}
        
        **🔍 Method:** {method_used}-based detection
        """)
    
    with col2:
        st.subheader("🤖 AI Response")
        st.success(f"""
        **💬 AI Response:**
        
        {ai_response}
        
        **🎭 Response style:** {response_style}
        
        **🧠 Generated by:** {llm_choice}
        """)
    
    # Audio response section
    st.subheader("🔊 AI Voice Response")
    
    if speech_audio:
        # Create audio player for AI response
        st.audio(speech_audio, format=audio_format)
        
        # Download button for audio
        st.download_button(
            label="⬇️ Download AI Response Audio",
            data=speech_audio,
            file_name=f"ai_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3",
            mime=audio_format
        )
    else:
        st.warning("⚠️ Text-to-Speech not available with current settings. Response shown as text above.")
        if tts_choice == "OpenAI TTS" and not openai_api_key:
            st.info("💡 Add your OpenAI API key in the sidebar to enable OpenAI TTS")

# Example conversations section
st.markdown("---")
st.subheader("💡 Example Conversations")

tab1, tab2, tab3, tab4 = st.tabs(["😢 Sad Input", "😡 Angry Input", "😄 Happy Input", "😰 Worried Input"])

with tab1:
    st.markdown("""
    **Example Sad Input:**
    - **🎤 User:** "I'm really struggling with this project and feeling overwhelmed..."
    - **😊 Detected:** Sad (0.89)
    - **🤖 AI Response:** "I'm sorry you're feeling overwhelmed. That sounds really challenging. Would it help to break the project down into smaller, more manageable steps? I'm here to help you work through this."
    """)

with tab2:
    st.markdown("""
    **Example Angry Input:**
    - **🎤 User:** "This software keeps crashing and I've lost all my work!"
    - **😊 Detected:** Angry (0.92)
    - **🤖 AI Response:** "I can hear how frustrated you are, and losing your work is incredibly annoying. Let's focus on recovering what we can and preventing this from happening again. Have you tried checking for auto-saved versions?"
    """)

with tab3:
    st.markdown("""
    **Example Happy Input:**
    - **🎤 User:** "I just got promoted at work and I'm so excited!"
    - **😊 Detected:** Happy (0.85)
    - **🤖 AI Response:** "That's fantastic news! Congratulations on your promotion! Your excitement is contagious. This must be the result of all your hard work paying off. What are you most looking forward to in your new role?"
    """)

with tab4:
    st.markdown("""
    **Example Worried Input:**
    - **🎤 User:** "I have a big presentation tomorrow and I'm really nervous about it..."
    - **😊 Detected:** Fear (0.78)
    - **🤖 AI Response:** "It's completely natural to feel nervous before an important presentation. That shows you care about doing well. Have you had a chance to practice it out loud? Sometimes rehearsing can help build confidence."
    """)

# Sidebar information
st.sidebar.markdown("---")
st.sidebar.title("ℹ️ About This Pipeline")
st.sidebar.info("""
This app creates a complete conversational AI pipeline:

1. **STT**: Converts your speech to text
2. **Emotion AI**: Detects your emotional state  
3. **LLM**: Generates appropriate responses
4. **TTS**: Speaks the response back to you

Perfect for therapy bots, customer service, educational assistants, and more!
""")

st.sidebar.title("🔧 Setup Instructions")
with st.sidebar.expander("Required packages"):
    st.code("""
pip install streamlit
pip install openai-whisper
pip install transformers
pip install torch
pip install librosa
pip install gtts
pip install requests
pip install matplotlib
""")

with st.sidebar.expander("API Keys (Optional)"):
    st.markdown("""
    **OpenAI API Key:**
    - Get from: https://platform.openai.com/
    - Enables: GPT responses + OpenAI TTS
    
    **Hugging Face API Key:**
    - Get from: https://huggingface.co/settings/tokens
    - Enables: Better LLM responses
    
    **Note:** App works without API keys using local models!
    """)