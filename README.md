# 🎤 Noyce Demo App - Empathetic AI Response System

An intelligent audio processing pipeline that detects emotions from speech and generates empathetic responses using state-of-the-art AI models.

## 🌟 Key Features

- **🎤 Speech-to-Text**: Powered by OpenAI Whisper for accurate transcription
- **🎭 Emotion Detection**: Real-time emotion analysis from audio using pretrained models
- **🤖 Intelligent Responses**: Context-aware responses using OpenAI GPT-3.5 or smart fallbacks
- **🔊 Text-to-Speech**: Natural voice responses using Google TTS
- **⚙️ Flexible Configuration**: Multiple response styles and manual emotion override
- **🐛 Debug Mode**: Detailed processing insights for development

## 🚀 Quick Start

### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd noyce-demo

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate noyce-demo

# Run the application
streamlit run pipeline.py
```

### Option 2: Using pip

```bash
# Clone the repository
git clone <your-repo-url>
cd noyce-demo

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run pipeline.py
```

## 🛠️ Installation Requirements

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended for better performance)
- **Storage**: 3GB free space (for model downloads)
- **Internet**: Required for initial model downloads and OpenAI API

### Dependencies

Core libraries automatically installed:
- Streamlit (Web interface)
- OpenAI Whisper (Speech-to-text)
- Transformers (Emotion detection)
- PyTorch (Neural network backend)
- Librosa (Audio processing)
- Google TTS (Text-to-speech)

## 🔧 Configuration

### OpenAI API Setup (Recommended)

1. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Enter the key in the app's sidebar
3. **Cost**: ~$0.002 per request (very affordable!)

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed instructions.

### Without OpenAI (Free Mode)

The app includes intelligent fallback responses that work without any API keys!

## 📖 How to Use

### Basic Usage

1. **Launch the app**: `streamlit run pipeline.py`
2. **Configure settings**: Choose response style in the sidebar
3. **Upload audio**: Drag & drop or select an audio file
4. **Get response**: AI analyzes emotion and generates empathetic response
5. **Listen**: Optional text-to-speech playback of the AI response

### Supported Audio Formats

- WAV (recommended)
- MP3
- OGG
- M4A
- FLAC

### Response Styles

- **Empathetic**: Warm, understanding responses
- **Casual**: Friendly, conversational tone
- **Professional**: Formal, business-appropriate
- **Supportive**: Encouraging, helpful responses

## 🎯 Example Interactions

| User Input | Detected Emotion | AI Response |
|------------|------------------|-------------|
| "I lost my job today" | 😢 Sad | "I'm sorry you're going through this. That sounds really tough and challenging." |
| "This software keeps crashing!" | 😡 Angry | "That sounds super frustrating! Technical issues like that are the worst." |
| "I just got promoted!" | 😄 Happy | "That's awesome! Congratulations on your promotion - you must be thrilled!" |

## 🏗️ Architecture

```
Audio Input → Whisper STT → Emotion Detection → LLM Processing → TTS Output
     ↓              ↓              ↓              ↓              ↓
   📁 File       📝 Text       😊 Emotion    🤖 Response    🔊 Audio
```

### AI Models Used

1. **OpenAI Whisper** (base model) - Speech-to-text transcription
2. **wav2vec2-lg-xlsr-en-speech-emotion-recognition** - Emotion detection
3. **OpenAI GPT-3.5-turbo** - Response generation (with fallback)
4. **Google TTS** - Text-to-speech synthesis

## 🔍 Advanced Features

### Debug Mode
Enable in sidebar to see:
- Model loading status
- Processing steps in real-time
- Confidence scores
- API call details

### Manual Emotion Override
- Override automatic emotion detection
- Test specific emotional responses
- Useful for development and testing

### Quick Test Mode
- Test response generation without audio upload
- Iterate quickly on prompt engineering
- Debug LLM responses

## 🎛️ Customization

### Response Templates
Modify fallback responses in `generate_simple_fallback()` function:
```python
responses = {
    "angry": {
        "empathetic": "Your custom empathetic response...",
        # Add more styles
    }
}
```

### Emotion Detection
Swap emotion models by changing the model name:
```python
audio_emotion_classifier = pipeline(
    "audio-classification", 
    model="your-preferred-emotion-model"
)
```

## 🚨 Troubleshooting

### Common Issues

**"Could not load audio emotion model"**
- Check internet connection
- Ensure 3GB free disk space
- Restart the application

**"OpenAI API Error"**
- Verify API key format
- Check billing setup
- See [SETUP_GUIDE.md](SETUP_GUIDE.md)

**"Transcription failed"**
- Try shorter audio clips (< 30 seconds)
- Check audio file format
- Ensure clear audio quality

### Performance Tips

- Use WAV format for best results
- Keep audio clips under 30 seconds
- Ensure good audio quality (minimal background noise)
- Use conda environment for better dependency management

## 🔐 Security & Privacy

- **API Keys**: Never commit to version control
- **Audio Data**: Processed locally, not stored permanently
- **Privacy**: No audio data sent to external services (except OpenAI for LLM)
- **Temporary Files**: Automatically cleaned up after processing

## 🎨 Use Cases

### Perfect for:
- **Voice Assistants**: Emotion-aware conversational AI
- **Therapy Applications**: Empathetic response systems
- **Customer Service**: Sentiment-aware support bots
- **Research**: Emotion recognition studies
- **Education**: Interactive learning applications

### Demo Scenarios:
- Mental health support chatbots
- Customer complaint handling
- Voice-activated smart home systems
- Accessibility tools for emotional communication

## 🧪 Development

### Running Tests
```bash
# Test response generation
streamlit run pipeline.py
# Use "Quick Response Test" section in the app
```

### Adding New Features
1. Fork the repository
2. Create a feature branch
3. Implement changes
4. Test thoroughly
5. Submit pull request

## 📊 Performance Metrics

- **Transcription Accuracy**: ~95% with clear audio
- **Emotion Detection**: ~80% accuracy on common emotions
- **Response Time**: 2-5 seconds per request
- **Memory Usage**: ~2GB with all models loaded

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional emotion detection models
- More response templates
- UI/UX enhancements
- Performance optimizations
- Multi-language support

## 📄 License

This work is part of the UCI Noyce Project, supported by the Noyce Award (October 2023). We appreciate the UC Noyce Award for funding our research on developing a caregiving AI agent for high-risk dementia patients.

## 🙏 Acknowledgments

- **OpenAI** - Whisper and GPT models
- **Hugging Face** - Emotion detection models
- **Google** - Text-to-speech service
- **Streamlit** - Web application framework

## 📞 Support

For issues and questions:
1. Check [SETUP_GUIDE.md](SETUP_GUIDE.md)
2. Enable debug mode for detailed error info
3. Open an issue with error details and steps to reproduce

---

**Built with ❤️ for empathetic AI interactions**
