import streamlit as st
import numpy as np
import pandas as pd
import random
from transformers import pipeline
import time
import base64
from io import BytesIO
import tempfile
import os
import wavio
import google.generativeai as genai
try:
    import sounddevice as sd
    from scipy.io.wavfile import write as write_wav
    AUDIO_SUPPORTED = True
except:
    AUDIO_SUPPORTED = False

# page configuration
st.set_page_config(
    page_title="Emotion-Aware Chatbot",
    page_icon="🐦‍🔥",
    layout="wide"
)

# CSS for responsive design
st.markdown("""
<style>

    .main {
        background-color: #f5f7ff;
        padding: 1rem;
        color: black;
    }
    .chat-container {
        max-height: 60vh;
        overflow-y: auto;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 0.8rem;
        background-color: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.8rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: row;
        align-items: flex-start;
        width: 100%;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    .chat-message.user {
        background-color: #e6f3ff;
        border: 1px solid #c8e1ff;
        margin-left: 10%;
        color: black;
    }
    .chat-message.bot {
        background-color: #f0f2f5;
        border: 1px solid #dfe1e5;
        margin-right: 10%;
        color: black;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
        flex-shrink: 0;
    }
    .chat-message .message {
        flex-grow: 1;
        word-break: break-word;
        font-size: 16px;
        line-height: 1.5;
        text-shadow: 0 0 1px rgba(0,0,0,0.1);
    }
    .quote-box {
        background-color: #f8f9fa;
        border-left: 5px solid #6c757d;
        padding: 1rem;
        margin: 1rem 0;
        font-style: italic;
        border-radius: 0.4rem;
    }
    .emotion-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        font-size: 0.75rem;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .input-container {
        background-color: white;
        padding: 1rem;
        border-radius: 0.8rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem auto;
        max-width: 800px;
    }
    .sidebar-content {
        padding: 1rem;
    }
    .mood-container {
        background-color: white;
        padding: 1rem;
        border-radius: 0.8rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem auto;
        max-width: 800px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: black;
        border: none;
        border-radius: 0.4rem;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #45a049;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .title-container {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .title-container h1 {
        margin: 0;
    }
    @media (max-width: 768px) {
        .chat-message {
            margin-left: 5%;
            margin-right: 5%;
        }
        .chat-message .avatar {
            width: 30px;
            height: 30px;
        }
        .chat-container {
            max-height: 50vh;
        }
        .input-container {
            margin: 0.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

emotion_map = {e: 0.1 for e in ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]}

def configure_gemini_api():
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        if "GEMINI_API_KEY" not in st.session_state:
            st.session_state.GEMINI_API_KEY = ""
        
        if st.session_state.GEMINI_API_KEY == "":
            st.sidebar.warning("Please enter your Gemini API key in the sidebar to enable response generation", icon="⚠️")
            st.session_state.GEMINI_API_KEY = st.sidebar.text_input("Enter Gemini API Key:", type="password")
            api_key = st.session_state.GEMINI_API_KEY
    
    if api_key:
        try:
            genai.configure(api_key=api_key)
            print(api_key)
            model = genai.GenerativeModel('gemini-2.0-flash')
            test_response = model.generate_content("Hello")
            
            if hasattr(test_response, 'text') and test_response.text.strip():
                return True
            else:
                st.sidebar.error("API key validation failed - empty response received. Please check your key.", icon="❌")
                return False
                
        except Exception as e:
            error_msg = str(e)
            if "403" in error_msg:
                st.sidebar.error("API key invalid or unauthorized. Please check your API key.", icon="❌")
            elif "404" in error_msg:
                st.sidebar.error("Model not found. The specified model 'gemini-2.0-flash' may not be available.", icon="❌")
            elif "429" in error_msg:
                st.sidebar.error("Rate limit exceeded. Please try again later.", icon="❌")
            else:
                st.sidebar.error(f"Error configuring Gemini API: {e}", icon="❌")
            return False
    return False
def configure_gemini_api():
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    
    if not api_key:
        if "GEMINI_API_KEY" not in st.session_state:
            st.session_state.GEMINI_API_KEY = ""
        
        if st.session_state.GEMINI_API_KEY == "":
            st.sidebar.warning("Please enter your Gemini API key in the sidebar to enable response generation", icon="⚠️")
            st.session_state.GEMINI_API_KEY = st.sidebar.text_input("Enter Gemini API Key:", type="password")
            api_key = st.session_state.GEMINI_API_KEY
    
    if api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash')
            test_response = model.generate_content("Hello")
            
            # Check if we got a valid response
            if hasattr(test_response, 'text') and test_response.text.strip():
                return True
            else:
                st.sidebar.error("API key validation failed - empty response received. Please check your key.", icon="❌")
                return False
                
        except Exception as e:
            error_msg = str(e)
            if "403" in error_msg:
                st.sidebar.error("API key invalid or unauthorized. Please check your API key.", icon="❌")
            elif "404" in error_msg:
                st.sidebar.error("Model not found. The specified model 'gemini-2.0-flash' may not be available.", icon="❌")
            elif "429" in error_msg:
                st.sidebar.error("Rate limit exceeded. Please try again later.", icon="❌")
            else:
                st.sidebar.error(f"Error configuring Gemini API: {e}", icon="❌")
            return False
    return False

def get_emotion_color(emotion):
    colors = {
        "happy": "#FFC107",
        "sad": "#2196F3",
        "angry": "#F44336",
        "fear": "#9C27B0",
        "surprise": "#FF9800",
        "disgust": "#4CAF50",
        "neutral": "#9E9E9E"
    }
    return colors.get(emotion, "#9E9E9E")
    
def record_audio(duration=5, sample_rate=16000):
    """Record audio using sounddevice and return as a file-like object"""
    if not AUDIO_SUPPORTED:
        st.error("Audio recording requires sounddevice and scipy. Please install with: pip install sounddevice scipy")
        return None
        
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_filename = temp_file.name
            
        # Record audio
        st.info(f"Recording for {duration} seconds...")
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        
        recording = (recording * 32767).astype(np.int16)
        write_wav(temp_filename, sample_rate, recording)
        
        return temp_filename
    except Exception as e:
        st.error(f"Error recording audio: {e}")
        return None

def get_emotion_emoji(emotion):
    emojis = {
        "happy": "😊",
        "sad": "😢",
        "angry": "😠",
        "fear": "😨",
        "surprise": "😲",
        "disgust": "😖",
        "neutral": "😐"
    }
    return emojis.get(emotion, "😐")

@st.cache_resource
def load_text_emotion_model():
    try:
        model = pipeline(
            task="text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None
        )
        return model
    except Exception as e:
        st.error(f"Error loading text emotion model: {e}")
        return None
        
@st.cache_resource
def load_audio_emotion_model():
    try:
        model = pipeline(
            task="audio-classification",
            model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        )
        return model
    except Exception as e:
        st.warning(f"Audio emotion model not loaded: {e}")
        return None

def detect_emotion_from_text(text, model):
    if not text.strip():
        return "neutral", {}
    
    try:
        if model:
            result = model(text)[0]
            emotion_scores = {item['label']: item['score'] for item in result}
            detected_emotion = max(emotion_scores, key=emotion_scores.get)
            return detected_emotion, emotion_scores
        else:
            emotions = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]
            detected_emotion = random.choice(emotions)
            return detected_emotion, {e: random.random() for e in emotions}
    except Exception as e:
        st.error(f"Error in text emotion detection: {e}")
        return "neutral", {}
        
def detect_emotion_from_audio(audio_path, model):
    """Detect emotion from audio using the audio emotion model"""
    try:
        if model and os.path.exists(audio_path):
            result = model(audio_path)
            # Get the emotion with the highest score
            detected_emotion = result[0]["label"]
            
            # Create emotion scores dictionary
            emotion_scores = {item["label"]: item["score"] for item in result}
            
            return detected_emotion, emotion_scores
        else:
            # Fallback if model fails or file doesn't exist
            emotions = ["happy", "sad", "angry", "fear", "surprise", "neutral"]
            detected_emotion = random.choice(emotions)
            return detected_emotion, {e: random.random() for e in emotions}
    except Exception as e:
        st.error(f"Error in audio emotion detection: {e}")
        return "neutral", {}

def generate_quote(emotion):
    """Generate a quote based on the detected emotion using Gemini API"""
    emotion_map = {
        "joy": "happy",
        "sadness": "sad",
        "anger": "angry",
        "disgust": "disgust",
        "fear": "fear",
        "surprise": "surprise",
        "neutral": "neutral"
    }
    mapped_emotion = emotion_map.get(emotion.lower(), "neutral")
    
    try:
        if configure_gemini_api():
            prompt = f"""Generate a short inspirational quote about {mapped_emotion} emotions. 
            The quote should be concise (1-2 sentences) and uplifting.
            Also provide the name of the author.
            Format your response as a JSON with two keys: "quote" and "author"
            For example: {{"quote": "The quote text here", "author": "Author Name"}}"""
            
            model = genai.GenerativeModel('gemini-2.0-flash')
            safety_settings = {
                "harassment": "block_none",
                "hate": "block_none",
                "sexual": "block_none",
                "dangerous": "block_none",
            }
            
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 200,
            }
            
            result = model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            response_text = result.text.strip()
            
            try:
                import json
                import re
                
                if "```json" in response_text:
                    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
                    if json_match:
                        response_text = json_match.group(1)
                
                quote_data = json.loads(response_text)
                
                if "quote" not in quote_data or "author" not in quote_data:
                    if "text" in quote_data and "attribution" in quote_data:
                        return {"quote": quote_data["text"], "author": quote_data["attribution"]}
                    else:
                        raise ValueError("Missing expected keys in response")
                        
                return quote_data
                
            except Exception as json_error:
                st.warning(f"JSON parsing error (will use fallback): {json_error}")
                quote_pattern = r'"quote"\s*:\s*"([^"]+)"'
                author_pattern = r'"author"\s*:\s*"([^"]+)"'
                
                quote_match = re.search(quote_pattern, response_text)
                author_match = re.search(author_pattern, response_text)
                
                if quote_match and author_match:
                    return {"quote": quote_match.group(1), "author": author_match.group(1)}
                lines = response_text.split("\n")
                quote_text = next((line for line in lines if line.strip() and not line.startswith('{')), 
                                "Every emotion has wisdom to share.")
                author_line = next((line for line in lines if "author" in line.lower() or "—" in line or "-" in line), "")
                if author_line:
                    author = author_line.split(":")[-1].strip().strip('"').strip("'")
                    if not author:
                        author = "Anonymous"
                else:
                    author = "Anonymous"
                    
                return {"quote": quote_text, "author": author}
        else:
            fallback_quotes = {
                "happy": {"quote": "Happiness is not something ready-made. It comes from your own actions.", "author": "Dalai Lama"},
                "sad": {"quote": "Even the darkest night will end and the sun will rise.", "author": "Victor Hugo"},
                "angry": {"quote": "For every minute you remain angry, you give up sixty seconds of peace of mind.", "author": "Ralph Waldo Emerson"},
                "fear": {"quote": "Fear is only as deep as the mind allows.", "author": "Japanese Proverb"},
                "surprise": {"quote": "Life is full of surprises and serendipity. Being open to unexpected turns in the road is an important part of success.", "author": "Todd Kashdan"},
                "disgust": {"quote": "Understanding is the first step to acceptance.", "author": "J.K. Rowling"},
                "neutral": {"quote": "The middle path is the way to wisdom.", "author": "Buddha"}
            }
            return fallback_quotes.get(mapped_emotion, {"quote": "The journey of a thousand miles begins with a single step.", "author": "Lao Tzu"})
    except Exception as e:
        st.error(f"Error generating quote: {e}")
        print(e)
        return {"quote": "The journey of a thousand miles begins with a single step.", "author": "Lao Tzu"}
    
def generate_response(emotion, user_message):
    """Generate a response based on emotion and user message using Gemini API"""
    emotion_map = {
        "joy": "happy",
        "sadness": "sad",
        "anger": "angry",
        "disgust": "disgust",
        "fear": "fear",
        "surprise": "surprise",
        "neutral": "neutral"
    }
    mapped_emotion = emotion_map.get(emotion.lower(), "neutral")
    
    try:
        if configure_gemini_api():
            prompt = f"""You are an empathetic AI assistant that responds with awareness of the user's emotional state.

            The user's current detected emotion is: {mapped_emotion}
            
            User message: {user_message}
            
            Please generate a thoughtful, supportive response that acknowledges their emotional state. 
            Keep your response concise (2-4 sentences) and empathetic.
            Your response should be appropriate for someone feeling {mapped_emotion}.
            Do not state that you're an AI or mention that you're responding based on detected emotions."""
            model = genai.GenerativeModel('gemini-2.0-flash')
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 200,
            }
            result = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            if hasattr(result, 'text'):
                response_text = result.text.strip()
                return response_text
            else:
                st.warning("Received invalid response format from Gemini API. Using fallback response.")
                return fallback_responses.get(mapped_emotion, "I appreciate your message. How can I help you further?")
            
        else:
            fallback_responses = {
                "happy": "I'm glad you're feeling happy! That's wonderful to hear.",
                "sad": "I understand you might be feeling down. Remember, it's okay to feel this way sometimes.",
                "angry": "I can sense you might be frustrated. Taking a deep breath can sometimes help.",
                "fear": "It's natural to feel afraid sometimes. Would you like to talk more about what's concerning you?",
                "surprise": "That sounds quite unexpected! How are you processing this surprise?",
                "disgust": "I understand something might be bothering you. Would you like to discuss what happened?",
                "neutral": "Thank you for sharing your thoughts with me."
            }
            return fallback_responses.get(mapped_emotion, "I appreciate your message. How can I help you further?")
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "I'm here to listen and chat with you. How can I help you today?"
    
def get_emotion_chart_html(emotion_scores):
    if not emotion_scores:
        return ""
    
    emotion_map = {
        "joy": "happy",
        "sadness": "sad",
        "anger": "angry",
        "disgust": "disgust", 
        "fear": "fear",
        "surprise": "surprise",
        "neutral": "neutral"
    }
    
    labels = list(emotion_scores.keys())
    values = [round(score * 100, 1) for score in emotion_scores.values()]
    colors = [get_emotion_color(emotion_map.get(label.lower(), "neutral")) for label in labels]
    
    html = f"""
    <div style="width:100%; height:180px;">
        <canvas id="emotionChart"></canvas>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        const ctx = document.getElementById('emotionChart');
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: {str(labels)},
                datasets: [{{
                    label: 'Emotion Score (%)',
                    data: {str(values)},
                    backgroundColor: {str(colors)},
                    borderColor: {str(colors)},
                    borderWidth: 1
                }}]
            }},
            options: {{
                indexAxis: 'y',
                scales: {{
                    x: {{
                        beginAtZero: true,
                        max: 100
                    }}
                }}
            }}
        }});
    </script>
    """
    return html

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_detected_emotion' not in st.session_state:
    st.session_state.last_detected_emotion = "neutral"
if 'emotion_scores' not in st.session_state:
    st.session_state.emotion_scores = {}
if 'audio_recording' not in st.session_state:
    st.session_state.audio_recording = None

gemini_available = configure_gemini_api()

if not AUDIO_SUPPORTED:
    st.warning("Voice recording support requires additional packages. Install with: `pip install sounddevice scipy`", icon="⚠️")
st.markdown('<div class="title-container">', unsafe_allow_html=True)
st.markdown(f'<h1>Emotion-Aware Chatbot</h1>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown("Chat with an AI that understands your emotions through text and voice!")

text_model = load_text_emotion_model()
audio_model = load_audio_emotion_model()

with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.header("About Emotion Aware Chatbot")
    st.write("""
    Emotion Aware ChatBot uses advanced AI to detect emotions from text and voice, providing personalized responses and inspirational quotes using Google's Gemini API.
    Along with that it uses fine tuned versions of pre-trained models like DistilBERT and word2vec to detect emotions from text and audio respectively.
    **Features:**
    - Real-time emotion detection
    - Text and voice input
    - AI-generated personalized responses
    - AI-generated inspirational quotes
    - Mood analysis
    """)
    
    st.header("Emotion Guide")
    emotions = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]
    for emotion in emotions:
        color = get_emotion_color(emotion)
        emoji = get_emotion_emoji(emotion)
        st.markdown(f"""
        <div class="emotion-badge" style="background-color: {color}; color: white;">
            {emoji} {emotion.capitalize()}
        </div>
        """, unsafe_allow_html=True)
    
    st.header("About Models")
    st.write("""
    This app uses the following models:
    - [j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) for text emotion detection
    - [ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition](https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition) for voice emotion detection
    - [Google Gemini API](https://ai.google.dev/) for generating personalized responses and inspirational quotes
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    debug_mode = st.checkbox("Show Debug Info")
    if debug_mode and 'debug_logs' in st.session_state:
        st.markdown("### Debug Logs")
        for log in st.session_state.debug_logs:
            st.text(log)

with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown("### Conversation")
    
    if not st.session_state.chat_history:
        st.info("Start the conversation below by typing a message or using voice input!")
    
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            emotion_emoji = get_emotion_emoji(chat["emotion"])
            emotion_color = get_emotion_color(chat["emotion"])
            st.markdown(f"""
            <div class="chat-message user">
                <div style="background-color: #0D8ABC; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; margin-right: 1rem;">
                    YO
                </div>
                <div class="message">
                    <b style="font-size: 18px;">You</b>
                    <p style="margin: 8px 0; color: #000;">{chat["message"]}</p>
                    <div style="margin-top: 5px;">
                        <span class="emotion-badge" style="background-color: {emotion_color}; color: white;">
                            {emotion_emoji} {chat["emotion"].capitalize()}
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot">
                <div style="background-color: #4CAF50; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; margin-right: 1rem;">
                    MC
                </div>
                <div class="message">
                    <b style="font-size: 18px;">Emotion Aware Chatbot</b>
                    <p style="margin: 8px 0; color: #000;">{chat["message"]}</p>
                    <div class="quote-box">"{chat["quote"]["quote"]}" — <b>{chat["quote"]["author"]}</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="input-container">', unsafe_allow_html=True)
st.markdown("### Conver your Emotion through words / voice")

input_method = st.radio("Choose input method:", ("Text", "Voice"), horizontal=True)

if input_method == "Text":
    user_input = st.text_input("Type your message:", key="text_input", 
                            placeholder="How are you feeling today?")
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Send", use_container_width=True) and user_input:
            with st.spinner("Analyzing emotion..."):
                emotion, emotion_scores = detect_emotion_from_text(user_input, text_model)
                st.session_state.last_detected_emotion = emotion
                st.session_state.emotion_scores = emotion_scores
                st.session_state.chat_history.append({
                    "role": "user",
                    "message": user_input,
                    "emotion": emotion
                })
                
                with st.spinner("Generating response..."):
                    bot_response = generate_response(emotion, user_input)
                    quote = generate_quote(emotion)
                    
                    st.session_state.chat_history.append({
                        "role": "bot",
                        "message": bot_response,
                        "quote": quote
                    })
                    st.rerun()
else:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("Click the button to record your voice message (5 seconds)")
    with col2:
        if st.button("Record Voice", use_container_width=True):
            if not AUDIO_SUPPORTED:
                st.error("Audio recording is not supported. Please install sounddevice and scipy packages.")
            else:
                with st.spinner("Recording for 5 seconds..."):
                    audio_file = record_audio(duration=5)
                
                if audio_file and os.path.exists(audio_file):
                    audio_bytes = open(audio_file, 'rb').read()
                    st.audio(audio_bytes, format="audio/wav")
                    
                    with st.spinner("Analyzing your voice emotion..."):
                        try:
                            emotion, emotion_scores = detect_emotion_from_audio(audio_file, audio_model)
                            st.session_state.last_detected_emotion = emotion
                            st.session_state.emotion_scores = emotion_scores
                            st.session_state.chat_history.append({
                                "role": "user",
                                "message": f"[Voice message: {emotion} detected]",
                                "emotion": emotion
                            })
                            bot_response = generate_response(emotion, "Voice message")
                            quote = generate_quote(emotion)
                            st.session_state.chat_history.append({
                                "role": "bot",
                                "message": bot_response,
                                "quote": quote
                            })
                            try:
                                os.remove(audio_file)
                            except:
                                pass
                                
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error processing audio: {e}")

st.markdown("### Quick Expression")
emotion_cols = st.columns(7)
emotions = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]
for i, emotion in enumerate(emotions):
    with emotion_cols[i]:
        emoji = get_emotion_emoji(emotion)
        if st.button(f"{emoji}", help=f"Express {emotion} feeling"):
            example_messages = {
                "happy": "I'm feeling really great today!",
                "sad": "I'm feeling a bit down today.",
                "angry": "I'm so frustrated with this situation.",
                "fear": "I'm worried about the upcoming deadline.",
                "surprise": "I can't believe what just happened!",
                "disgust": "That was really unpleasant to experience.",
                "neutral": "Just checking in to say hello."
            }
            user_input = example_messages[emotion]
            
            if emotion == "happy": emotion_map["joy"] = 0.9
            elif emotion == "sad": emotion_map["sadness"] = 0.9
            elif emotion == "angry": emotion_map["anger"] = 0.9
            elif emotion == "fear": emotion_map["fear"] = 0.9
            elif emotion == "surprise": emotion_map["surprise"] = 0.9
            elif emotion == "disgust": emotion_map["disgust"] = 0.9
            else: emotion_map["neutral"] = 0.9
            
            st.session_state.last_detected_emotion = emotion
            st.session_state.emotion_scores = emotion_map
            st.session_state.chat_history.append({
                "role": "user",
                "message": user_input,
                "emotion": emotion
            })
            
            with st.spinner("Generating response..."):
                bot_response = generate_response(emotion, user_input)
                quote = generate_quote(emotion)
                
                st.session_state.chat_history.append({
                    "role": "bot",
                    "message": bot_response,
                    "quote": quote
                })
                st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
if st.session_state.chat_history:
    st.markdown('<div class="mood-container">', unsafe_allow_html=True)
    st.markdown("### Current Mood Analysis")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        emotion = st.session_state.last_detected_emotion
        color = get_emotion_color(emotion)
        emoji = get_emotion_emoji(emotion)
        
        st.markdown(f"""
        <div style="
            width: 100px; 
            height: 100px; 
            border-radius: 50%; 
            background-color: {color}; 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            color: white; 
            font-size: 40px;
            font-weight: bold;
            margin: 0 auto;
        ">
            {emoji}
        </div>
        <div style="text-align: center; margin-top: 10px; font-weight: bold;">
            {emotion.capitalize()}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.session_state.emotion_scores:
            st.components.v1.html(get_emotion_chart_html(st.session_state.emotion_scores), height=200)
        
        recommendations = {
            "happy": "Keep spreading that positivity! Try some creative activities to harness this energy.",
            "sad": "Consider some self-care activities or reaching out to a friend. Remember it's okay to feel down sometimes.",
            "angry": "Deep breathing exercises might help reduce tension. Try a calming activity like walking or journaling.",
            "fear": "Write down your worries to gain perspective. Breaking concerns into smaller parts can make them more manageable.",
            "surprise": "Embrace the unexpected! Surprise often leads to new opportunities and perspectives.",
            "disgust": "Reflect on what triggered this feeling to understand your boundaries and values better.",
            "neutral": "This balanced state is great for focused work, planning, or decision-making."
        }
        
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;">
            <h4>Recommendation</h4>
            <p>{recommendations.get(emotion, "Keep exploring your emotions!")}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown("""
---
**Note**: This is a demo application using Hugging Face models for emotion detection and response generation. Not intended for clinical use.
""")