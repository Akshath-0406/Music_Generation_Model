import os
import json
import numpy as np
import tensorflow.keras as keras
import music21 as m21
from music21 import environment
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH

# Setting an environment variable to allow Music21 to use MuseScore as default music notation software.
mscore_path = os.environ['MUSIC21_MSCORE']
environment.set("musicxmlPath", mscore_path)

class MelodyGen:
    def __init__(self, model_path = "model.h5"):
        # Constructor that initializes the model
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)
        
        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)
        
        self._start_symbols = ["/"] * SEQUENCE_LENGTH
        
    
    def gen_melody(self, seed, num_steps, max_seq_length, temperature):
        # Create seed with start symbol
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed
        
        # Map seed to integer
        seed = [self._mappings[symbol] for symbol in seed]
        
        for i in range(num_steps):
            # Limiting the seed to maximum sequence length
            seed = seed[-max_seq_length:]
            
            # One-hot encoding the seed
            onehot_seed = keras.utils.to_categorical(seed, num_classes = len(self._mappings))
            # Dimensions of seed should be changed to (1, maximum sequence length, vocabulary size)
            onehot_seed = onehot_seed[np.newaxis, ...]
            
            # Making a prediction using the new seed
            probabilities = self.model.predict(onehot_seed)[0]
            # Each symbol has a probability of happening. We need topick out the most probable output
            output_int = self._sample_with_temperature(probabilities, temperature)
            
            # Updating the seed
            seed.append(output_int)
            
            # Mapping integer to our encoding
            output_symbol = [k for k,v in self._mappings.items() if v == output_int][0]
            
            # Check if we are at the end of melody
            if output_symbol == "/":
                break
            
            # Updating the melody
            melody.append(output_symbol)
            
        return melody
    
    
    def _sample_with_temperature(self, probabilities, temperature):
        # Remodelling the probability distribution table by choosing different temperature values
        # Temperature closer to 0 would make the distribution more deterministic.
        # Temperature closer to 1 would make the distribution more unpredictable.
        predictions = np.log(probabilities) / temperature
        
        # Applying softmax function to remodel probability distribution
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))
        
        choices = range(len(probabilities))
        
        # Sampling the output based on the new probability distribution
        index = np.random.choice(choices, p=probabilities)
        
        return index
    
    
    def save_melody(self, melody, step_duration = 0.25, format = "midi", file_name = "mel.mid"):
        # Converts the melody from gen_melody into a MIDI file
        
        # Create a music21 stream
        stream = m21.stream.Stream()
        
        start_symbol = None
        step_counter = 1
        
        # Parse all symbols in the melody and create note/rest objects
        for i, symbol in enumerate(melody):
            # Handle cases in which we have a note or rest or when melody is one iteration away from ending
            if symbol != "_" or i + 1 == len(melody):
                # Ensuring that we're dealing with notes/rests beyond the first one
                if start_symbol is not None:
                    
                    quarter_length_duration = step_duration * step_counter
                    
                    # Handling a rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)
                    
                    # Handling a note
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)
                    
                    stream.append(m21_event)
                    
                    # Reset the step counter
                    step_counter = 1
                    
                start_symbol = symbol
            
            # Handle case in which we have a prolongation sign "_"
            else:
                step_counter += 1
        
        # Write the m21 stream to a midi file
        stream.write(format, file_name)
        return stream


if __name__ == "__main__":
    mg = MelodyGen()
    seed = "67 _ 67 _ 67 _ _ 65 64 _ 64 _ 64 _ _"
    seed2 = "67 _ _ _ _ _ 65 _ 64 _ 62 _ 60 _ _ _"
    melody = mg.gen_melody(seed, 500, SEQUENCE_LENGTH, 0.4)
    print(melody)
    stream = mg.save_melody(melody)
    stream.show()