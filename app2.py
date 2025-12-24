# ============================
# Imports
# ============================

# Flask is used to create the backend API server
from flask import Flask, request, jsonify

# CORS allows frontend / browser / Postman / Colab to call this API
from flask_cors import CORS

# SpeechRecognition converts audio (WAV) → text (Hindi)
import speech_recognition as sr

# PyTorch is used to run the language model
import torch

# HuggingFace Transformers for tokenizer, model, and pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# OS utilities (file handling)
import os

# Used to create temporary files for uploaded audio
import tempfile

# Used to call ffmpeg for audio conversion
import subprocess


# ============================
# Flask App Setup
# ============================

# Create Flask application
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing
CORS(app)


# ============================
# Model Setup
# ============================

# Print message so we know model loading has started
print("\nLoading Hindi car assistant model...", flush=True)

# HuggingFace model name (small & CPU-friendly)
MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"

# Force CPU for maximum stability in Colab
device = "cpu"
dtype = torch.float32

print(f"Using device: {device}", flush=True)

# Load tokenizer (converts text → tokens)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load language model weights
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=dtype
).to(device)

# Create a text-generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1  # -1 means CPU
)


# ============================
# Prompt Template
# ============================

# This prompt controls the behavior of the AI
# It tells the model:
# 1. It is a car voice assistant
# 2. It must reply briefly
# 3. It must reply in Hindi
alpaca_prompt_template = """You are a helpful car voice assistant.
Reply briefly in Hindi.

User command:
{}

Assistant response:
"""


# ============================
# Helper Functions
# ============================

def ai_car_response(user_command: str) -> str:
    """
    Sends user command to the LLM and returns the generated Hindi response.
    """

    # Insert user command into the prompt
    prompt = alpaca_prompt_template.format(user_command)

    # Generate text using the model
    outputs = pipe(
        prompt,
        max_new_tokens=80,     # Limit response length (important for CPU)
        do_sample=True,
        temperature=0.7,       # Controls creativity
        top_p=0.9
    )

    # Extract only the assistant's response
    text = outputs[0]["generated_text"]
    return text.split("Assistant response:")[-1].strip()


def car_action_logic(user_command: str):
    """
    Rule-based logic for instant car actions.
    This avoids calling the LLM for simple commands.
    """

    if "गर्मी" in user_command:
        return "ठीक है, मैं एसी चालू कर रहा हूँ।"

    if "ठंड" in user_command:
        return "ठीक है, मैं एसी बंद कर रहा हूँ।"

    if "खिड़की" in user_command and "खोल" in user_command:
        return "ठीक है, मैं खिड़की खोल रहा हूँ।"

    if "खिड़की" in user_command and "बंद" in user_command:
        return "ठीक है, मैं खिड़की बंद कर रहा हूँ।"

    # If no rule matches, return None
    return None


# ============================
# API Endpoint
# ============================

@app.route("/api/voice-command", methods=["POST"])
def handle_voice_command():
    """
    Main API endpoint:
    1. Receives audio from client
    2. Converts audio to WAV
    3. Performs Hindi speech recognition
    4. Generates AI response
    5. Returns JSON output
    """

    print("\n📥 Request received", flush=True)

    webm_path = None
    wav_path = None

    try:
        # Ensure audio file exists in request
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files["audio"]
        recognizer = sr.Recognizer()

        # Save uploaded audio as WebM
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            audio_file.save(tmp.name)
            webm_path = tmp.name

        # Prepare WAV path
        wav_path = webm_path.replace(".webm", ".wav")

        print("🔄 Converting audio to WAV", flush=True)

        # Convert WebM → WAV using ffmpeg
        # - Mono channel
        # - 16kHz sample rate
        # - Remove silence
        # - Increase volume
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", webm_path,
                "-ac", "1",
                "-ar", "16000",
                "-af", "silenceremove=1:0:-50dB,volume=2",
                wav_path
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        # Load WAV audio for speech recognition
        with sr.AudioFile(wav_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.6)
            audio_data = recognizer.record(source)

        print("🟡 Calling Google Speech API", flush=True)

        # Convert speech → Hindi text
        user_command_text = recognizer.recognize_google(
            audio_data, language="hi-IN"
        )

        print("🧠 User said:", user_command_text, flush=True)

        # First try rule-based response (fast)
        rule_response = car_action_logic(user_command_text)

        if rule_response:
            car_response = rule_response
            print("⚙️ Rule-based response used", flush=True)
        else:
            # If no rule matches, use LLM
            print("🤖 Using LLM fallback", flush=True)
            car_response = ai_car_response(user_command_text)

        # Send final JSON response
        return jsonify({
            "user_command": user_command_text,
            "car_response": car_response
        })

    # Speech not understood
    except sr.UnknownValueError:
        return jsonify({
            "error": "मैं समझ नहीं पाया। कृपया धीरे और साफ़ बोलें।"
        }), 400

    # Google Speech API failure
    except sr.RequestError as e:
        print("❌ Google Speech API error:", e, flush=True)
        return jsonify({
            "error": "Speech service unavailable"
        }), 503

    # Any other unexpected error
    except Exception as e:
        print("🔥 ERROR:", repr(e), flush=True)
        return jsonify({"error": str(e)}), 500

    # Always clean up temporary files
    finally:
        for f in [webm_path, wav_path]:
            if f and os.path.exists(f):
                os.remove(f)
        print("🧹 Temp files cleaned", flush=True)


# ============================
# Run Server
# ============================

if __name__ == "__main__":
    # Start Flask server
    # - No debug (prevents restart loops)
    # - No reloader (important in Colab)
    app.run(
        host="0.0.0.0",
        port=5001,
        debug=False,
        use_reloader=False
    )
