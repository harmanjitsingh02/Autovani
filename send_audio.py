import requests
import os

url = "http://127.0.0.1:5001/api/voice-command"

audio_path = "audio/Hinjawadi_fixed.wav"   # ✅ use converted wav

if not os.path.exists(audio_path):
    raise FileNotFoundError(f"Audio file not found: {audio_path}")

with open(audio_path, "rb") as f:
    files = {
        # filename should be simple, not full path
        "audio": ("fixed.wav", f, "audio/wav")
    }

    try:
        response = requests.post(
            url,
            files=files,
            timeout=30
        )

        print("Status code:", response.status_code)
        print("Response JSON:", response.json())

    except requests.exceptions.RequestException as e:
        print("❌ Request failed:", e)
