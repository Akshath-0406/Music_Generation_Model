# Music_Generation_Model
A music generation model built using Long Short Term Memory Recurrent Neural Network (LSTM RNN).  
  
Dataset: http://www.esac-data.org/  

The dataset is first pre-processed by converting the songs to a format that can be understood by our model. This is done by mapping song notes into integers and making sure that the model understands our map.
We then train the model on the pre-processed dataset and save it after training.
The saved dataset then can be run to accept a seed (user input song notes which acts as the start for the generated song), and then generates the rest of the song by choosing the most probable next note.
