from TTS.api import TTS

class VoiceGenerator:
    def __init__(self):
        # Initialize the TTS model
        self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

    def generate_voice(self, lyrics, output_file):
        """Generate voice audio from lyrics."""
        self.tts.tts_to_file(text=lyrics, file_path=output_file)

# Example usage
if __name__ == "__main__":
    lyrics = "In the midnight sky, birds cry out alone..."
    output_file = 'generated_voice.wav'
    generator = VoiceGenerator()
    generator.generate_voice(lyrics, output_file)
