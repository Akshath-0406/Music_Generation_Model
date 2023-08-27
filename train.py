import tensorflow.keras as keras
from preprocess import gen_train_seq, SEQUENCE_LENGTH
from preprocess import vocab_size

OUTPUT_UNITS = vocab_size()
NUM_UNITS = [256]
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 64
MODEL_PATH = "model.h5"


def build_model(output_units, num_units, loss, learning_rate):
    # Creating model architecture
    input_layer = keras.layers.Input(shape = (None, output_units))
    x = keras.layers.LSTM(num_units[0])(input_layer)
    x = keras.layers.Dropout(0.2)(x)
    
    output_layer = keras.layers.Dense(output_units, activation = "softmax")(x)
    
    model = keras.Model(input_layer, output_layer)
    
    # Compiling the model
    model.compile(loss = loss,
                  optimizer = keras.optimizers.Adam(learning_rate = learning_rate),
                  metrics = ["accuracy"])
    model.summary()
    return model


def training(output_units = OUTPUT_UNITS, num_units = NUM_UNITS, loss = LOSS, learning_rate = LEARNING_RATE):
    # Generating training sequence using preprocess.py
    inputs, targets = gen_train_seq(SEQUENCE_LENGTH)
    
    # Build the model
    model = build_model(output_units, num_units, loss, learning_rate)
    
    # Train the model
    model.fit(inputs, targets, epochs = EPOCHS, batch_size = BATCH_SIZE)
    
    # Save the model
    model.save(MODEL_PATH)


if __name__ == "__main__":
    training()