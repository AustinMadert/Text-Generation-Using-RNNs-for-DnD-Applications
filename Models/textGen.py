# -*- coding: utf-8 -*-
#!pip install tensorflow-gpu==2.0.0-beta1
import subprocess
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class Keras_Text_Generator(object):

    def __init__(self, batch_size=64, buffer_size=10000, checkpoint_dir = './training_checkpoints'):
        self.batch_size=batch_size
        self.buffer_size=buffer_size
        self.model=tf.keras.Sequential()
        self.dataset=None
        self.checkpoint_dir = checkpoint_dir
        checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt_{epoch}")
        self.checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True)


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

        # Create character to number (and vice versa) lookups
        self.char2idx = {j:i for i, j in enumerate(self.vocab)}
        self.idx2char = np.array(self.vocab)

        # Transform text string into integer representation
        self.text_as_int = np.array([self.char2idx[char] for char in self.text_string])

        return None


    def _split_input_target(self, chunk):
        '''
        Takes a string and creates an input string and target string.
        '''

        input_text = chunk[:-1]
        target_text = chunk[1:]

        return input_text, target_text


    def _create_rolling_sequences(self):
        '''
        Takes text_string and creates seq_length training vectors by incrementing 
        through the text_string one character at a time.
        '''

        dataX = []
        dataY = []
        for i in range(0, self.length_of_data - self.seq_length):
            seq_in = self.text_string[i:i + self.seq_length]
            seq_out = self.text_string[i + 1:i + self.seq_length + 1]
            dataX.append([self.char2idx[char] for char in seq_in])
            dataY.append([self.char2idx[char] for char in seq_out])
        
        return (dataX, dataY)


    def _create_dataset(self):
        """Takes length of data and desired training instance length and creates
        training instances of desired length and concatenates them into a dataset.        
        
        Arguments:
            None
        
        Returns:
            None
        """

        # Comments needed
        if self.rolling_sequences:
            rolling_dataset = tf.data.Dataset.from_tensor_slices(self._create_rolling_sequences())
            self.dataset = rolling_dataset.shuffle(self.buffer_size).batch(self.batch_size, 
                        drop_remainder=True)
        else:
            char_dataset = tf.data.Dataset.from_tensor_slices(self.text_as_int)
            sequences = char_dataset.batch(self.seq_length + 1, drop_remainder=True)
            dataset = sequences.map(self._split_input_target)
            self.dataset = dataset.shuffle(self.buffer_size).batch(self.batch_size, 
                        drop_remainder=True)

        return None


    def load_and_create_dataset(self, filename, seq_length=100, rolling_sequences=True):
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
        self.text_string = self._load_data(filename)

        # Store for later use
        self.seq_length = seq_length
        self.length_of_data = len(self.text_string)
        self.rolling_sequences = rolling_sequences

        # Create model character vocabulary
        self.vocab = sorted(set(self.text_string))
        self.vocab_size = len(self.vocab)

        print(f'Length of text: {self.length_of_data} characters')
        print(f'Unique characters: {self.vocab_size}')

        self._vectorize_text()
        self._create_dataset()

        print('Dataset successfully created.')
        
        return None


    def add_layer_to_model(self, layer, **kwargs):
        '''
        Takes layer type and number of units and adds new layer to model.
        '''
        if len(self.model.layers) == 0:
            self.model.add(layer(batch_input_shape=[self.batch_size, None], **kwargs))
        else:
            self.model.add(layer(**kwargs))

        return None


    def _loss(self, labels, logits):
        '''
        Takes labels and logits from model and returns tf crossentropy loss function
        '''
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


    def compile_model(self, optimizer='adam'):
        '''
        Take loss and optimizer inputs then compiles and displays model.
        '''
        self.model.compile(optimizer=optimizer, loss=self._loss)
        self.model.summary()

        return None

    
    def fit_model(self, epochs=10):
        '''
        Take number of epochs and fits model using class dataset and checkpoints.
        '''

        # Adjust steps_per_epoch input based on data generation technique used
        if self.rolling_sequences:
            steps_per_epoch = (self.length_of_data - self.seq_length) // self.batch_size
        else:
            steps_per_epoch = (self.length_of_data / self.seq_length) // self.batch_size

        # Train the model
        history = self.model.fit(self.dataset, epochs=epochs,\
            callbacks=[self.checkpoint_callback], steps_per_epoch=steps_per_epoch)

        tf.train.latest_checkpoint(self.checkpoint_dir)

        return None


    def load_model_from_checkpoint(self):
        '''
        Loads model weights from existing checkpoint. Class model needs to have
        same architecture as the trained model.
        '''

        self.model.load_weights(tf.train.latest_checkpoint(self.checkpoint_dir))
        self.model.build(tf.TensorShape([1, None]))
        self.model.summary()

        return None


    def generate_text(self, start_string='<<start>>', temperature=1.0, num_generate=2000):
        """Once a model is trained or loaded, takes a start string and iteratively
        generates the specified number of output characters. Temperature controls
        the randomness of characters generated, or how suprising the characters
        are. Higher temperature means more surprising based on the dataset.
        
        Keyword Arguments:
            start_string {str} -- the initial inputs to the text generated. (default: {'<<start>>'})
            temperature {float} -- Controls how surprising generated text is. (default: {1.0})
            num_generate {int} -- Number of characters to generate. (default: {2000})
        
        Returns:
            Output Text [str] -- The generated text based on input.
        """
        input_eval = [self.char2idx[char] for char in start_string]
        input_eval = tf.expand_dims(input_eval, 0)
        
        text_generated = []
        
        self.model.reset_states()
        for _ in range(num_generate):
            predictions = self.model(input_eval)
            predictions = tf.squeeze(predictions, 0)
            
            predictions = predictions/temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0]
            
            input_eval = tf.expand_dims([predicted_id], 0)
            
            text_generated.append(self.idx2char[predicted_id])
            
        self.output = (start_string + ''.join(text_generated))

        return self.output