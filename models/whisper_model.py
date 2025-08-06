"""from faster_whisper import WhisperModel

def load_whisper_model(model_size="base", device="cpu", compute_type="int8"):
    return WhisperModel(model_size, device=device, compute_type=compute_type)"""

from faster_whisper import WhisperModel

def get_whisper_model():
    """Initialize and return a WhisperModel instance with fallback logic."""
    try:
        model = WhisperModel("base", device="cpu", compute_type="int8")
        print("Loaded Faster Whisper base model")
        return model
    except Exception as e:
        print(f"Failed to load base model: {e}")
        try:
            model = WhisperModel("tiny", device="cpu", compute_type="int8")
            print("Loaded Faster Whisper tiny model")
            return model
        except Exception as e2:
            print(f"Failed to load tiny model: {e2}")
            # Last resort - try medium model
            model = WhisperModel("medium", device="cpu", compute_type="int8")
            print("Loaded Faster Whisper medium model")
            return model

