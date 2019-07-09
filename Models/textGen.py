# -*- coding: utf-8 -*-
#!pip install tensorflow-gpu==2.0.0-beta1
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


    def _vectorize_text(self, data_string):
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


    def _create_dataset(self):
        """Takes length of data and desired training instance length and creates
        training instances of desired length and concatenates them into a dataset.        
        
        Arguments:
            length_of_data {[int]} -- actual length of text data string
            seq_length {[int]} -- desired length of training instances
        
        Returns:
            None
        """
        # Create a tensorflow Dataset object 
        char_dataset = tf.data.Dataset.from_tensor_slices(self.text_as_int)
        # Use batch method to create training instances of desired size
        sequences = char_dataset.batch(self.seq_length + 1, drop_remainder=True)
        # Create training instances and labels
        dataset = sequences.map(self._split_input_target)
        self.dataset = dataset.shuffle(self.buffer_size).batch(self.batch_size, drop_remainder=True)

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
        # Store seq_length for later use
        self.seq_length = seq_length

        # Load the data
        data_string = self._load_data(filename)
        self.length_of_data = len(data_string)

        # Create model character vocabulary
        self.vocab = sorted(set(data_string))
        self.vocab_size = len(self.vocab)

        print(f'Length of text: {self.length_of_data} characters')
        print(f'Unique characters: {self.vocab_size}')

        self._vectorize_text(data_string)
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
        steps_per_epoch = (self.length_of_data - self.seq_length)//self.batch_size

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
        for i in range(num_generate):
            predictions = self.model(input_eval)
            predictions = tf.squeeze(predictions, 0)
            
            predictions = predictions/temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
            
            input_eval = tf.expand_dims([predicted_id], 0)
            
            text_generated.append(self.idx2char[predicted_id])
            
        self.output = (start_string + ''.join(text_generated))

        return self.output