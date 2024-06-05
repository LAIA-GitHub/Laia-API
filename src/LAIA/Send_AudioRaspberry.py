import requests

def send_audio_to_raspberry(audio_path):
    url = 'http://<raspberry_pi_ip>:5000/upload-audio'
    with open(audio_path, 'rb') as f:
        files = {'audio': f}
        response = requests.post(url, files=files)
        return response.status_code

def send_transcription_to_raspberry(transcription):
    url = 'http://<raspberry_pi_ip>:5000/transcription'
    response = requests.post(url, json={"transcription": transcription})
    return response.status_code