"""
voice.py
--------
Voice Mode — Speech-to-Text input and Text-to-Speech output.

Uses:
  - SpeechRecognition (Google Web Speech API) for transcription
  - gTTS (Google Text-to-Speech) for spoken answers
  - pydub for audio processing

The flow:
  User speaks → record audio → transcribe → RAG pipeline → synthesize speech → play back
"""

import io
import os
import tempfile
import base64
from pathlib import Path


# ─── Text-to-Speech ───────────────────────────────────────────────────────────

def text_to_speech(text: str, lang: str = "en") -> bytes:
    """
    Convert text to speech using gTTS.
    Returns raw MP3 bytes that can be played in a browser via Streamlit.

    Args:
        text : The text to speak (LLM answer)
        lang : Language code (default 'en')

    Returns:
        MP3 audio as bytes
    """
    from gtts import gTTS

    # Clean text for TTS — remove markdown symbols
    clean = (
        text
        .replace("**", "")
        .replace("*", "")
        .replace("#", "")
        .replace("`", "")
        .replace("•", "")
        .replace("⚠️", "Warning:")
        .replace("✓", "Correct answer:")
    )

    tts = gTTS(text=clean, lang=lang, slow=False)
    mp3_buffer = io.BytesIO()
    tts.write_to_fp(mp3_buffer)
    mp3_buffer.seek(0)
    return mp3_buffer.read()


def audio_to_base64(audio_bytes: bytes) -> str:
    """Encode audio bytes as base64 for embedding in HTML audio tag."""
    return base64.b64encode(audio_bytes).decode("utf-8")


def make_audio_html(audio_bytes: bytes, autoplay: bool = True) -> str:
    """
    Return an HTML <audio> element with the encoded MP3.
    This can be rendered via st.markdown(html, unsafe_allow_html=True).
    """
    b64 = audio_to_base64(audio_bytes)
    autoplay_attr = "autoplay" if autoplay else ""
    return f"""
    <audio controls {autoplay_attr} style="width:100%;margin-top:8px;border-radius:8px;">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """


# ─── Speech-to-Text ───────────────────────────────────────────────────────────

def transcribe_audio_file(audio_bytes: bytes, file_format: str = "wav") -> str:
    """
    Transcribe audio bytes to text using Google Web Speech API.

    Args:
        audio_bytes : Raw audio file bytes (WAV or WebM)
        file_format : 'wav' or 'webm'

    Returns:
        Transcribed text string
    """
    import speech_recognition as sr

    recognizer = sr.Recognizer()

    with tempfile.NamedTemporaryFile(suffix=f".{file_format}", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        with sr.AudioFile(tmp_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data)
        return text

    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        raise RuntimeError(f"Speech recognition service unavailable: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def get_voice_input_js() -> str:
    """
    Returns JavaScript for browser-based microphone recording.
    Uses the Web Speech API (SpeechRecognition) available in Chrome/Edge.
    This runs client-side and posts the transcript back via Streamlit's
    component communication.
    """
    return """
    <script>
    function startVoiceRecognition(callbackId) {
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            alert('Voice recognition is not supported in this browser. Please use Chrome or Edge.');
            return;
        }
        
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();
        
        recognition.lang = 'en-US';
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;
        recognition.continuous = false;
        
        recognition.onstart = () => {
            document.getElementById('voice-status').textContent = '🔴 Listening...';
            document.getElementById('voice-btn').disabled = true;
        };
        
        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            document.getElementById('voice-transcript').value = transcript;
            document.getElementById('voice-status').textContent = '✅ Got it! ' + transcript;
            document.getElementById('voice-btn').disabled = false;
            
            // Trigger Streamlit to pick up the value
            document.getElementById('voice-transcript').dispatchEvent(new Event('input', { bubbles: true }));
        };
        
        recognition.onerror = (event) => {
            document.getElementById('voice-status').textContent = '❌ Error: ' + event.error;
            document.getElementById('voice-btn').disabled = false;
        };
        
        recognition.onend = () => {
            if (document.getElementById('voice-status').textContent === '🔴 Listening...') {
                document.getElementById('voice-status').textContent = '⏸ No speech detected';
            }
            document.getElementById('voice-btn').disabled = false;
        };
        
        recognition.start();
    }
    </script>
    
    <div style="display:flex;flex-direction:column;gap:8px;">
        <div style="display:flex;align-items:center;gap:10px;">
            <button id="voice-btn" onclick="startVoiceRecognition()" 
                style="padding:10px 20px;background:#7c6fff;color:white;border:none;
                       border-radius:20px;cursor:pointer;font-size:14px;font-weight:600;">
                🎤 Speak Question
            </button>
            <span id="voice-status" style="font-size:13px;color:#6b7280;">Click to start</span>
        </div>
        <input id="voice-transcript" type="hidden" value="">
    </div>
    """
