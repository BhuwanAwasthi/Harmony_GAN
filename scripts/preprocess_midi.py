import os
import pretty_midi
import numpy as np
from tqdm import tqdm

def collect_midi_files(root_dir, max_files=500):
    """Collect MIDI files from the directory, limited to max_files."""
    midi_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.mid') or file.lower().endswith('.midi'):
                midi_files.append(os.path.join(root, file))
                if len(midi_files) >= max_files:
                    return midi_files
    return midi_files

def midi_to_piano_roll(midi_path, fs=100):
    """Convert a MIDI file to a piano roll matrix."""
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        piano_roll = midi_data.get_piano_roll(fs=fs)
        piano_roll = (piano_roll > 0).astype(np.float32)
        return piano_roll
    except Exception as e:
        print(f"Error processing {midi_path}: {e}")
        return None

def preprocess_midi(midi_files, output_file):
    """Preprocess MIDI files and save the piano rolls."""
    piano_rolls = []
    max_length = 0

    # First pass to find maximum length
    for midi_path in midi_files:
        piano_roll = midi_to_piano_roll(midi_path)
        if piano_roll is not None:
            max_length = max(max_length, piano_roll.shape[1])

    # Second pass to pad sequences
    for midi_path in tqdm(midi_files, desc="Processing MIDI files"):
        piano_roll = midi_to_piano_roll(midi_path)
        if piano_roll is not None:
            padded_piano_roll = np.pad(
                piano_roll,
                ((0, 0), (0, max_length - piano_roll.shape[1])),
                mode='constant'
            )
            piano_rolls.append(padded_piano_roll)

    # Convert to NumPy array
    piano_rolls_array = np.array(piano_rolls)

    # Save the piano rolls
    np.save(output_file, piano_rolls_array)
    print(f"Processed {len(piano_rolls)} MIDI files saved to {output_file}")

if __name__ == "__main__":
    # Path to your MIDI files
    midi_root_dir = '/content/drive/MyDrive/lakh_midi_clean'  # Update this path
    output_file = 'data/midi/processed_piano_rolls.npy'

    # Collect and preprocess the MIDI files
    midi_files = collect_midi_files(midi_root_dir, max_files=500)
    preprocess_midi(midi_files, output_file)
