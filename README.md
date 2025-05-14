# Sentiment-Aware Chatbot 🤖 💭

A Streamlit-based chatbot application that uses advanced AI to detect emotions from text and voice input, providing personalized responses and inspirational quotes tailored to the user's emotional state.

![Emotion-Aware Chatbot](https://i.imgur.com/example.png)

## Table of Contents
- [Features](#features)
- [Demo](#demo)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [API Keys](#api-keys)
- [How It Works](#how-it-works)
- [Model Information](#model-information)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Multi-modal Emotion Detection**: Analyzes emotions from both text and voice inputs
- **Interactive UI**: Clean, responsive interface with real-time feedback
- **Emotion Visualization**: Displays emotion analysis with charts
- **Personalized Responses**: AI-generated responses based on detected emotions
- **Inspirational Quotes**: Provides contextually relevant quotes for emotional support
- **Voice Recording**: Record and analyze voice messages directly in the app
- **Quick Expression Buttons**: Express common emotions with a single click
- **Mood Analysis**: Tracks and visualizes your emotional state throughout the conversation

## Demo

https://www.canva.com/design/DAGnbDhCScg/pPDMYLBtpQB9Rg8FvVO3EA/view?utm_content=DAGnbDhCScg&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h5188703e8c

## Requirements

- Python 3.8+
- Streamlit
- HuggingFace Transformers
- PyTorch
- Google Gemini API key
- (Optional) SoundDevice and SciPy for voice recording functionality

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/emotion-aware-chatbot.git
cd emotion-aware-chatbot
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. (Optional) For voice recording support:
```bash
pip install sounddevice scipy
```

## Usage

1. Set up your Google Gemini API key:
   - Create a `.streamlit/secrets.toml` file with your API key:
   ```toml
   GEMINI_API_KEY = "your_api_key_here"
   ```
   - Alternatively, you can enter your API key directly in the app's sidebar

2. Run the application:
```bash
streamlit run chatbot.py
```

3. Open your browser and navigate to the URL provided by Streamlit (typically http://localhost:8501)

4. Start chatting! You can:
   - Type messages in the text input
   - Record voice messages (if supported)
   - Use quick expression buttons to share emotions

## API Keys

This application uses the Google Gemini API for generating responses and quotes. You'll need to:

1. Sign up for a Google Gemini API key at [https://ai.google.dev/](https://ai.google.dev/)
2. Add your API key to the app as described in the Usage section
3. Or add it in the .streamlit folder as secret.toml with GEMINI_API_KEY = '<API_KEY>'

## 🧠 How It Works

### Emotion Detection
- **Text Analysis**: Uses DistilRoBERTa-based model to classify emotions in text
- **Voice Analysis**: Employs wav2vec2 model to detect emotions from audio recordings

### Response Generation
- The detected emotion informs the prompt to Google's Gemini API
- Responses are tailored to be supportive and appropriate for the user's emotional state

### Quote Generation
- Inspirational quotes are generated based on the detected emotion
- Each quote is paired with an author attribution

## 🤖 Model Information

The application uses the following models:
- **Text Emotion Detection**: [j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
- **Voice Emotion Detection**: [ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition](https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition)
- **Response Generation**: Google Gemini 2.0 Flash model via the Gemini API

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Troubleshooting

### Common Issues

**Issue**: Audio recording not working
**Solution**: Ensure you have installed the optional dependencies with `pip install sounddevice scipy`

**Issue**: Model loading errors
**Solution**: Check your internet connection, as models are downloaded from HuggingFace on first run

**Issue**: API key issues
**Solution**: Verify your API key is correct and properly configured in the secrets.toml file

### Debug Mode

Enable debug mode in the sidebar to see detailed logs that can help troubleshoot issues.

---

For voice support, also add:
```
sounddevice>=0.4.5
scipy>=1.8.0
```
