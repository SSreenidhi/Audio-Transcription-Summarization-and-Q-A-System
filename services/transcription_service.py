def transcribe_audio(whisper_model, audio_file, task='translate'):
    segments, info = whisper_model.transcribe(audio_file, task=task)
    return " ".join([segment.text for segment in segments])