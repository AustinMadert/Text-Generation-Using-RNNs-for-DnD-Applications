# -*- coding: utf-8 -*-
#!pip install tensorflow-gpu==2.0.0-beta1
import tensorflow as tf
import numpy as np
import os
import time

class rnn_network(object):

    def __init__(self, batch_size=64, buffer_size=10000):
        self.batch_size=batch_size
        self.buffer_size=buffer_size
        self.model=tf.keras.Sequential()
        self.dataset=None


    def _load_data(self, filename):
        '''
        Opens and reads text file. Returns string.
        '''
        with open(filename, 'r') as f:
            data = f.read()

        return data


    def _vectorize_text(self):
        '''
        Creates indexes to transform characters to integers representation and back.
        Transforms text string into integer representation.
        '''
        self.char2idx = {j:i for i, j in enumerate(self.vocab)}
        self.idx2char = np.array(self.vocab)

        # Transform text string into integer representation
        self.text_as_int = np.array([self.char2idx[char] for char in data_string])

        return None


    def _split_input_target(self, chunk):
        '''
        Takes a string and creates an input string and target string.
        '''
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text


    def _create_dataset(self, length_of_data, seq_length):
        """Takes length of data and desired training instance length and creates
        training instances of desired length and concatenates them into a dataset.        
        
        Arguments:
            length_of_data {[int]} -- actual length of text data string
            seq_length {[int]} -- desired length of training instances
        
        Returns:
            None
        """
        examples_per_epoch = length_of_data//seq_length

        # Create a tensorflow Dataset object 
        char_dataset = tf.data.Dataset.from_tensor_slices(self.text_as_int)
        # Use batch method to create training instances of desired size
        sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)
        # Create training instances and labels
        self.dataset = sequences.map(self._split_input_target)

        return None


    def load_and_create_dataset(self, filename, seq_length=100):
        """Method designed to load a text file and create a training dataset. Data
        are loaded, then vectorized, and finally, the data is batched into training
        instances.
        
        Arguments:
            filename {string} -- location of text file
        
        Keyword Arguments:
            seq_length {int} -- desired length of training instances (default: {100})
        
        Returns:
            None
        """
        # Load the data
        data_string = _load_data(filename)
        length_of_data = len(data_string)

        # Create model character vocabulary
        self.vocab = sorted(set(data_string))
        self.vocab_size = len(self.vocab)

        print(f'Length of text: {length_of_data} characters')
        print(f'Unique characters: {self.vocab_size}')

        self._vectorize_text()
        self._create_dataset(length_of_data, seq_length)

        print('Dataset successfully created.')
        
        return None
        
        


    def build_model(vocab_size, embedding_dim, rnn_units=200, batch_size):
      model = tf.keras.Sequential([
          tf.keras.layers.Embedding(vocab_size, embedding_dim, 
                                  batch_input_shape=[batch_size, None]),
          tf.keras.layers.LSTM(rnn_units, 
                              return_sequences=True,
                              stateful=True,
                              recurrent_initializer='glorot_uniform'),
          tf.keras.layers.LSTM(rnn_units, 
                              return_sequences=True,
                              stateful=True,
                              recurrent_initializer='glorot_uniform'),
          tf.keras.layers.Dense(vocab_size)
          
      ])
      return model

    model = build_model(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        batch_size=BATCH_SIZE
    )

    def open_file(filename):
      f = open('filename')
      monster_string = f.read()
      f.close()
      







dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

dataset

vocab_size = len(vocab)

embedding_dim = 256

rnn_units = 1000

count = []
for i in dataset.take(-1):
  count.append(i)
  
print(len(count), len(count) * 64)



model.summary()

for ex_batch_input, ex_batch_target in dataset.take(1):
  ex_batch_pred = model(ex_batch_input)
  print(ex_batch_pred.shape,"# (batch_size, sequence_length, vocab_size)")

sampled_indices = tf.random.categorical(ex_batch_pred[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
sampled_indices

print('Input: \n', repr(''.join(idx2char[ex_batch_input[0]])))
print()
print('Output: \n', repr(''.join(idx2char[sampled_indices])))

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_loss = loss(ex_batch_target, ex_batch_pred)
print("Prediction shape: ", ex_batch_pred.shape, " # (batch_size, sequence_length, vocab_size)")
print("Scalar loss: ", example_loss.numpy().mean())

model.compile(optimizer='adam', loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS = 30

history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

tf.train.latest_checkpoint(checkpoint_dir)

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

model.summary()

def generate_text(model, start_string, num_generate=2000):
  
  input_eval = [char2idx[char] for char in start_string]
  input_eval = tf.expand_dims(input_eval, 0)
  
  text_generated = []
  
  temperature = 1.0
  
  model.reset_states()
  for i in range(num_generate):
    predictions = model(input_eval)
    predictions = tf.squeeze(predictions, 0)
    
    predictions = predictions/temperature
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
    
    input_eval = tf.expand_dims([predicted_id], 0)
    
    text_generated.append(idx2char[predicted_id])
    
  return (start_string + ''.join(text_generated))

output = generate_text(model, start_string='<<start>>')
print(output)

def build_gru_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
      tf.keras.layers.Embedding(vocab_size, embedding_dim, 
                               batch_input_shape=[batch_size, None]),
      tf.keras.layers.GRU(rnn_units, 
                          return_sequences=True,
                          stateful=True,
                          recurrent_initializer='glorot_uniform'),
      tf.keras.layers.GRU(rnn_units, 
                          return_sequences=True,
                          stateful=True,
                          recurrent_initializer='glorot_uniform'),
      tf.keras.layers.GRU(rnn_units, 
                          return_sequences=True,
                          stateful=True,
                          recurrent_initializer='glorot_uniform'),
      tf.keras.layers.Dense(vocab_size)
      
  ])
  return model

gru_model = build_gru_model(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE
)

gru_model.compile(optimizer='adam', loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './gru_training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS = 40

history = gru_model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

gru_model = build_gru_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

gru_model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

gru_model.build(tf.TensorShape([1, None]))

gru_model.summary()

print(generate_text(gru_model, start_string='<<start>>'))

seq_length = 100
dataX = []
dataY = []
for i in range(0, len(monster_string) - seq_length):
  seq_in = monster_string[i:i + seq_length]
  seq_out = monster_string[i + 1:i + seq_length + 1]
  dataX.append([char2idx[char] for char in seq_in])
  dataY.append([char2idx[char] for char in seq_out])

print(len(dataX), len(dataY), len(dataX[0]), len(dataY[0]))

new_dataset = tf.data.Dataset.from_tensor_slices((dataX, dataY))

BATCH_SIZE = 64

BUFFER_SIZE = 10000

new_dataset = new_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

new_dataset

vocab_size = len(vocab)

embedding_dim = 256

rnn_units = 100

def build_gru_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
      tf.keras.layers.Embedding(vocab_size, embedding_dim, 
                               batch_input_shape=[batch_size, None]),
      tf.keras.layers.GRU(rnn_units, 
                          return_sequences=True,
                          stateful=True,
                          recurrent_initializer='glorot_uniform'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(vocab_size)
      
  ])
  return model

gru_model = build_gru_model(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE
)

gru_model.compile(optimizer='adam', loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './gru_training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

gru_model.summary()

EPOCHS = 1

history = gru_model.fit(new_dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

gru_model = build_gru_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

gru_model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

gru_model.build(tf.TensorShape([1, None]))

gru_model.summary()

print(generate_text(gru_model, start_string='<<start>>'))

