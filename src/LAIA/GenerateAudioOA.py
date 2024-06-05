from pathlib import Path
from openai import OpenAI
# Load environment variables
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

def generate_audio_with_OpenAI(text):
    
    try:
        # Specify the directory for saving audio files
        save_dir = Path(__file__).parent / "docs/audiooutput"
        save_dir.mkdir(parents=True, exist_ok=True)

        speech_file_path = save_dir / "speech.mp3"

        response = client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=text
        )
        print("Audio generated successfully")
        response.stream_to_file(speech_file_path)
        return str(speech_file_path)
    except Exception as e:
        print(f"Failed to generate audio with OpenAI: {e}")
        return None         



