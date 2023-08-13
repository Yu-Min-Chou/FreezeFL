import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import metrics

from utils.language_utils import letter_to_vec, word_to_indices

AUTOTUNE = tf.data.experimental.AUTOTUNE

class ClientModel:
    def __init__(self, lr, batch_size, seq_len, num_classes, num_hidden):
        self.lr = lr
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.num_hidden = num_hidden

        self.model = self.create_model()
        self.model.summary()

        self.model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr),
                        loss = tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    
    def create_model(self):
        x = inputs = layers.Input(shape=(self.seq_len), dtype=tf.int32)
        x = tf.keras.layers.Embedding(self.num_classes, 8, input_length=self.seq_len)(x)
        x = tf.keras.layers.LSTM(units=self.num_hidden, return_sequences=True, dtype=tf.float32)(x)
        x = tf.keras.layers.LSTM(units=self.num_hidden, dtype=tf.float32)(x)
        x = tf.keras.layers.Dense(self.num_classes, activation='sigmoid')(x)

        return models.Model(inputs, x, name='RNN')

    def train(self, dataset_train, num_local_epochs):
        self.model.fit(dataset_train, epochs = num_local_epochs)

        return self.get_model_weight()
        
    def test(self, dataset_test):
        loss_result = 0
        accu_result = 0

        results = self.model.evaluate(dataset_test, batch_size=self.batch_size)
        loss_result = results[0]
        accu_result = results[1]

        return loss_result, accu_result*100

    def set_model_weight(self, model_weight):
        self.model.set_weights(model_weight)

    def get_model_weight(self):
        return self.model.get_weights()

    def get_model_size(self):
        trainableParams = np.sum([np.prod(v.get_shape()) for v in self.model.trainable_weights])
        nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in self.model.non_trainable_weights])
        # unit: MB
        return (trainableParams + nonTrainableParams) * (64 / 8) / 1024**2

    def process_data(self, raw_data):
        raw_x_data = np.array([word_to_indices(word) for word in raw_data['x']])
        raw_y_data = np.array([letter_to_vec(c) for c in raw_data['y']])
        dataset = tf.data.Dataset.from_tensor_slices((raw_x_data, raw_y_data))
        dataset = dataset.shuffle(50000, reshuffle_each_iteration=True)
        dataset = dataset.batch(self.batch_size, drop_remainder=False).prefetch(AUTOTUNE)

        return dataset
