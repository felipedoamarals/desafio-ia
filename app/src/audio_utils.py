from typing import Optional

def transcribe_wav(audio_path: str, model_size: str = "base") -> Optional[str]:
    """Transcreve o WAV usando Whisper. Retorna texto ou None em caso de falha."""
    try:
        import whisper 
        model = whisper.load_model(model_size)
        result = model.transcribe(audio_path, language="pt")
        return result.get("text", "").strip()
    except Exception as e:
        print(f"Falha na transcrição: {e}")
        return None