from models.lyrics_generator import LyricsGenerator
from models.music_generator import MusicGeneratorGAN
from models.voice_generator import VoiceGenerator
from scripts.merge_audio import merge_audio
import numpy as np
import pretty_midi
import soundfile as sf
import fluidsynth
import os

def save_piano_roll_as_midi(piano_roll, output_file, fs=100):
    """Convert piano roll to MIDI file."""
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
    for note_number in range(piano_roll.shape[0]):
        indices = np.where(piano_roll[note_number, :] > 0)
        for idx in indices[0]:
            note = pretty_midi.Note(
                velocity=100,
                pitch=note_number,
                start=idx / fs,
                end=(idx + 1) / fs
            )
            instrument.notes.append(note)
    pm.instruments.append(instrument)
    pm.write(output_file)

def synthesize_midi_to_audio(midi_file, output_file, soundfont_path='soundfont.sf2'):
    """Convert MIDI file to audio using Fluidsynth."""
    fs = fluidsynth.Synth()
    fs.start()
    sfid = fs.sfload(soundfont_path)
    fs.program_select(0, sfid, 0, 0)
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            fs.noteon(0, note.pitch, note.velocity)
            fs.noteoff(0, note.pitch)
    samples = fs.get_samples(44100)
    sf.write(output_file, samples, 44100, subtype='PCM_16')
    fs.delete()

def main():
    # Step 1: Generate lyrics
    user_input = input("Enter your request: ")
    lyrics_generator = LyricsGenerator()
    lyrics = lyrics_generator.generate_lyrics(user_input)
    print("\nGenerated Lyrics:\n")
    print(lyrics)

    # Step 2: Generate music
    music_generator = MusicGeneratorGAN()
    noise = np.random.normal(0, 1, (1, 100))
    piano_roll = music_generator.generator.predict(noise).reshape(128, 500)

    # Save piano roll as MIDI
    midi_file = 'generated_music.mid'
    save_piano_roll_as_midi(piano_roll, midi_file)

    # Synthesize MIDI to audio
    music_audio_file = 'generated_music.wav'
    synthesize_midi_to_audio(midi_file, music_audio_file, soundfont_path='soundfont.sf2')

    # Step 3: Generate voice
    voice_generator = VoiceGenerator()
    voice_audio_file = 'generated_voice.wav'
    voice_generator.generate_voice(lyrics, voice_audio_file)

    # Step 4: Merge audio files
    output_song_file = 'final_song.wav'
    merge_audio(music_audio_file, voice_audio_file, output_song_file)

    print(f"\nFinal song saved as {output_song_file}")

if __name__ == "__main__":
    main()
