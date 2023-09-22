# Music_Generation_Model
A music generation model built using Long Short Term Memory Recurrent Neural Network (LSTM RNN).  
  
**Dataset:** http://www.esac-data.org/  

**Requirements:** The following libraries need to be installed using pip install: os, json, numpy, tensorflow, music21.  

## Data Pre-Processing:  
Several steps are used to pre-process the data. The data is first in kern format. Each kern file is loaded onto the program. Each song is then checked if there are any invalid notes or durations. Each song is then transposed to a standard scale (A minor or C major). Now the songs are encoded to represent time-series data. This is done by converting notes to their MIDI values, rests to 'r' and extended durations with '_'. After encoding each song, they are all put into a single file. The encoded songs are then mapped to integer values so it can be feeded to the model during training.  

## Training:  
We build an LSTM model with four layers. The first layer is the input layer which accepts inputs. The input layer is connected to the LSTM layer. The LSTM layer is then connected to the Dropout layer to prevent overfitting. The dropout layer is connected to the output layer, which has a softmax activation function that classifies the output to one of the notes (or rests or duration). The model is then compiled and returned.  
This model is trained using the pre-processed data and then saved.  

## Output:  
First we take an input 'seed' from the user. A seed is the first few notes of the song that has to be generated. The user can give any notes and then the model generates the rest of the song until it reaches the sequence limit. The model then finds the probability of each note occuring after the previous note. LSTM also takes long-term dependency into consideration. The model then outputs the highest probable note. This procedure keeeps repeating until the sequence limit is reached.
