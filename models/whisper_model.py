from faster_whisper import WhisperModel

def load_whisper_model(model_size="base", device="cpu", compute_type="int8"):
    return WhisperModel(model_size, device=device, compute_type=compute_type)
