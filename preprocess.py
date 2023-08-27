import os
import json
import music21 as m21
import numpy as np
import tensorflow.keras as keras

DATASET_PATH = "deutschl/erk"
SAVE_PATH = "dataset"
SINGLE_FILE_DATASET = "file_dataset"
SEQUENCE_LENGTH = 64
MAPPING_PATH = "mapping.json"

ACCEPTABLE_DURATIONS = [
    0.25, # 16th note
    0.5, # 8th note
    0.75, # Dotted 8th note
    1.0, # Quarter note
    1.5, # Dotted quarter note
    2, # Half note
    3, # Dotted half note
    4 # Whole note
    ]


def load_songs(dataset_path):
    songs = []
    # Going through all files in dataset and loading them with music21.
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == "krn":
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs


def check_durations(song, acceptable_durations):
    # Flatten the song into a list and check if all notes and rests have acceptible durations.
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True


def transpose(song):
    #Transposing a song to C Major or A Minor.
    #This is done so that the model learns only on C Major and A Minor scales instead of all 24 scales.
    
    # Directly getting key from the song.
    parts = song.getElementsByClass(m21.stream.Part)
    measure = parts[0].getElementsByClass(m21.stream.Measure) # Getting measures of first part.
    key = measure[0][4] # Getting key from the first measure in the first path.
    
    # Estimating the key using music21.
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")
    
    # Calculating the interval (distance) needed for transposition.
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))
    
    # Transposing the song using the calculated interval.
    transposed_song = song.transpose(interval)
    return transposed_song


def encode(song, time_step = 0.25):
    # Converts a song into a time-series representation. Makes use of MIDI notation.
    # We make use of '_' to signify how long a pitch or a rest is being played for.
    # The song is represented using steps, where each step is either a pitch or a rest or '_'.
    # Each step represents the "minimum duration", which is 1/4th of a quarter length.
    encoded_song = []
    for event in song.flat.notesAndRests:
        # For handling notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi
        # For handling rests
        elif isinstance(event, m21.note.Rest):
            symbol = "r"
        
        steps = int(event.duration.quarterLength/time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")
    
    # Type casting the time-series representation list into a string.
    encoded_song = " ".join(map(str, encoded_song))
    return encoded_song


def preprocessing(dataset_path):

    # Loading the songs from the dataset.
    print("Loading songs...")
    songs = load_songs(dataset_path)
    print(f"Loaded {len(songs)} songs.")
    
    for i, song in enumerate(songs):
        
        # Filtering out songs that have unacceptable durations.
        if not check_durations(song, ACCEPTABLE_DURATIONS):
            continue
        
        # Transposing songs to Cmaj/Amin.
        song = transpose(song)
        
        # Encode songs with music time-series representation.
        encoded_song = encode(song)
        
        # Save songs to text file.
        save_path = os.path.join(SAVE_PATH, str(i))
        with open(save_path, "w") as fp:
            fp.write(encoded_song)


def load(file_path):
    # To load the encoded songs onto the string in single_file_dataset function
    with open(file_path, "r") as fp:
        song = fp.read()
    return song


def single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    # Loading all encoded songs, add delimiters between each song, and save it in a string.
    song_delimiter = "/ " * sequence_length
    songs = ""
    
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + song_delimiter
    
    # Remove the empty character from the end of the string.
    songs = songs[:-1]
    
    with open(file_dataset_path, "w") as fp:
        fp.write(songs)
    return songs


def mapping(songs, mapping_path):
    # Create a json file that maps the symbols in the song dataset onto integers.
    mappings = {}
    
    # Identifying the vocabulary (Unique instances that represent similar symbols).
    songs = songs.split()
    vocabulary = list(set(songs))
    
    # Creating the mapping.
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i
    
    # Saving the vocabulary to a json file.
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent = 5)


def songs_to_int(songs):
    # We convert the songs string from single file dataset into intengers using the mapping we created.
    int_songs = []
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)
    
    songs = songs.split()
    for symbol in songs:
        int_songs.append(mappings[symbol])
    return int_songs


def gen_train_seq(sequence_length):
    # Generating a traning sequence to train the LSTM model.
    songs = load(SINGLE_FILE_DATASET)
    int_songs = songs_to_int(songs)
    
    inputs = []
    targets = []
    
    num_seq = len(int_songs) - sequence_length
    for i in range(num_seq):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])
    
    # One-hot encoding the sequence
    vocabulary_size = len(set(int_songs))
    inputs = keras.utils.to_categorical(inputs, num_classes = vocabulary_size)
    targets = np.array(targets)
    return inputs, targets


def vocab_size():
    # This is for train.py to access vocabulary size, as number of output units = vocabulary size
    songs = load(SINGLE_FILE_DATASET)
    int_songs = songs_to_int(songs)
    vocabulary_size = len(set(int_songs))
    return vocabulary_size


def main():
    preprocessing(DATASET_PATH)
    songs = single_file_dataset(SAVE_PATH, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    mapping(songs, MAPPING_PATH)
    inputs, targets = gen_train_seq(SEQUENCE_LENGTH)


if __name__ == "__main__":
    main()



# To delete the MXML files from MuScore: C:\Users\aksha\AppData\Local\Temp\music21