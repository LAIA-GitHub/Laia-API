from openai import OpenAI 

def transcribe_audio(client, file_path):
    """ Transcribe audio using OpenAI's Whisper model. """
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )
    return transcription.text

    
   