import numpy as np
import time
import gc
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import metrics

from baseline_constants import SYSTEM_PARAMS
from utils.language_utils import letter_to_vec, word_to_indices

IMAGE_SIZE = 28
CHANNEL_SIZE = 1
AUTOTUNE = tf.data.experimental.AUTOTUNE

class ClientModel:
    def __init__(self, lr, batch_size, seq_len, num_classes, num_hidden):
        self.lr = lr
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.num_hidden = num_hidden

        self.model_list = []
        self.model = None
        self.create_models()
    
    def create_model(self, p):
        x = inputs = layers.Input(shape=(self.seq_len), dtype=tf.int32)
        x = tf.keras.layers.Embedding(self.num_classes, 8, input_length=self.seq_len)(x)
        x = tf.keras.layers.LSTM(units=int(self.num_hidden*p), return_sequences=True, dtype=tf.float32)(x)
        x = tf.keras.layers.LSTM(units=int(self.num_hidden*p), dtype=tf.float32)(x)
        x = tf.keras.layers.Dense(self.num_classes, activation='sigmoid')(x)

        return models.Model(inputs, x, name='RNN')

    def create_models(self):
        num_tier = SYSTEM_PARAMS['NUM_TIER']
        for i in range(1, num_tier+1):
            model = self.create_model(i/num_tier)
            model.summary()
            model.compile(optimizer ='adam',
                        loss = tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
            self.model_list.append(model)

        self.model = self.model_list[num_tier-1]

    def dropout_train(self,dataset_train, num_local_epochs, OD_index):

        self.model = self.model_list[OD_index]

        self.model.fit(dataset_train, epochs = num_local_epochs)

        return self.model.get_weights()

    def test(self, dataset_test):
        self.model = self.model_list[SYSTEM_PARAMS['NUM_TIER']-1]
        loss_result = 0
        accu_result = 0

        results = self.model.evaluate(dataset_test, batch_size=self.batch_size)

        loss_result = results[0]
        accu_result = results[1]

        return loss_result, accu_result*100

    def set_model_weight(self, model_weight):
        for i in range(len(self.model_list)):
            self.model_list[i].set_weights(model_weight[i])

    def get_model_weight(self):
        return [self.model_list[i].get_weights() for i in range(len(self.model_list))]

    def get_model_size(self, OD_index):
        trainableParams = np.sum([np.prod(v.get_shape()) for v in self.model_list[OD_index].trainable_weights])
        nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in self.model_list[OD_index].non_trainable_weights])
        # unit: MB

        return (trainableParams + nonTrainableParams) * (64 / 8) / 1024**2

    def process_data(self, raw_data):
        raw_x_data = np.array([word_to_indices(word) for word in raw_data['x']])
        raw_y_data = np.array([letter_to_vec(c) for c in raw_data['y']])
        dataset = tf.data.Dataset.from_tensor_slices((raw_x_data, raw_y_data))
        dataset = dataset.shuffle(50000, reshuffle_each_iteration=True)
        dataset = dataset.batch(self.batch_size, drop_remainder=False).prefetch(AUTOTUNE)

        return dataset
