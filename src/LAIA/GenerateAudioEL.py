import requests
from elevenlabs import client
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from dotenv import load_dotenv
import os
from pathlib import Path
load_dotenv()

client = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_KEY")
)

voice_id = "Sarah"

def generate_audio_with_elevenlabs(text:str) -> str:
    
    try:
        # Call the ElevenLabs generate function directly
        # Specify the directory for saving audio files
        #save_dir = Path(__file__).parent.parent / "docs/audiooutput"
        #save_dir.mkdir(parents=True, exist_ok=True)

        audio_path = "docs/audiooutput/elevenlabs_audio.wav"

        response = client.text_to_speech.convert(text=text, voice_id="EXAVITQu4vr4xnSDxMaL", model_id="eleven_multilingual_v2")
        
        # Handle the generator response
        with open(audio_path, 'wb') as audio_file:
            for chunk in response:
                if chunk:
                    audio_file.write(chunk)
        
        print(f"{audio_path}:Audio generated successfully")
        return str(audio_path)
    except Exception as e:
        print(f"Failed to generate audio with ElevenLabs: {e}")
        return None
  




    
