import subprocess

def merge_audio(music_file, voice_file, output_file):
    """Merge music and voice audio files using FFmpeg."""
    command = f'ffmpeg -i "{music_file}" -i "{voice_file}" -filter_complex amix=inputs=2:duration=longest "{output_file}"'
    subprocess.call(command, shell=True)

# Example usage
if __name__ == "__main__":
    music_file = 'generated_music.wav'
    voice_file = 'generated_voice.wav'
    output_file = 'final_song.wav'
    merge_audio(music_file, voice_file, output_file)
