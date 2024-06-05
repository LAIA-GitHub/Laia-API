from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders.merge import MergedDataLoader
import TranscribeAudio
import SupaBase
import local_data_loader
import CreateVector
import ModifyingPrompt
import os
import GenerateAudioEL

# recieve audio file, POST LoadingState2
def main():
    # Load environment variables
    load_dotenv()

    # Create a client instance
    client = OpenAI()

    # Define the path to your audio file
    audio_path = "docs/audiorecord/TestRecord.wav"
    
    # Get the transcription
    transcription_text = TranscribeAudio.transcribe_audio(client, audio_path)
    
    # Print the transcription
    print("Transcription:\n", transcription_text)

    # Setup Supabase client
    supabase_client = SupaBase.setup_supabase_client()

    SupaBase.fetch_data_from_database_and_save(supabase_client)

    #SupaBase.subscribe_to_changes(supabase_client)

    # Load documents from local folder
    local_docs = local_data_loader.load_local_documents("docs/opendata")

    # Fetch data from Supabase
    database_docs = local_data_loader.load_local_documents("docs/inputdata")

    # Combine documents from local and database
    combined_docs = [*local_docs, *database_docs]
    #print(combined_docs)
    # Combine documents from local and database
    #combined_docs = local_docs + live_data

    # Create and load vector store
    vector_store = CreateVector.create_vector_store(combined_docs)
    chain = ModifyingPrompt.create_chain(vector_store)

    # Invoke the chain
    response = chain.invoke({
        "input": transcription_text,
        "context": vector_store
    })
    print(response['answer'])


    # detect Language of transcript and return

    # Call Generate Audio (if catalan AINA, else Elevenlabs), Send Audio to Raspberry Pi
    response = response['answer']
    GenerateAudioEL.generate_audio_with_elevenlabs(response)






if __name__ == "__main__":
    main()
