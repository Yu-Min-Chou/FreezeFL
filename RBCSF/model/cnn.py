import numpy as np
import time
import gc
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import metrics

from baseline_constants import SYSTEM_PARAMS

IMAGE_SIZE = 28
CHANNEL_SIZE = 1
AUTOTUNE = tf.data.experimental.AUTOTUNE

class ClientModel:
    def __init__(self, lr, batch_size, num_classes):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.lr = lr
        self.model = self.create_model()
        self.model.summary()

        self.model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr, momentum = 0.9, decay=1e-4),
                        loss = tf.keras.losses.SparseCategoricalCrossentropy())
    
    def create_model(self):
        x = inputs = layers.Input((IMAGE_SIZE*IMAGE_SIZE*CHANNEL_SIZE))
        x = tf.keras.layers.Reshape((IMAGE_SIZE, IMAGE_SIZE, CHANNEL_SIZE))(x)
        x = tf.keras.layers.Conv2D(32, 5, padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = tf.keras.layers.Conv2D(64, 5, padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)

        return models.Model(inputs, x, name='CNN')

    def fair_train(self, dataset_train, num_local_epochs):
        self.model.fit(dataset_train, epochs = num_local_epochs)

        return self.get_model_weight() 

    def test(self, dataset_test):
        sparse_categorical_accuracy = metrics.SparseCategoricalAccuracy()
        loss_result = 0
        accu_result = 0

        sparse_categorical_accuracy.reset_states()
        for x, y in dataset_test:
            logits = self.model(x)
            sparse_categorical_accuracy.update_state(y_true=y, y_pred=logits)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=logits)
            loss_result += tf.reduce_mean(loss)
        
        loss_result /= len(dataset_test)
        accu_result = sparse_categorical_accuracy.result() * 100

        return loss_result.numpy(), accu_result.numpy()

    def set_model_weight(self, model_weight):
        self.model.set_weights(model_weight)

    def get_model_weight(self):
        return self.model.get_weights()

    def get_model_size(self):
        trainableParams = np.sum([np.prod(v.get_shape()) for v in self.model.trainable_weights])
        nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in self.model.non_trainable_weights])
        # unit: MB

        return (trainableParams + nonTrainableParams) * (32 / 8) / 1024**2

    def process_data(self, raw_data):
        dataset = tf.data.Dataset.from_tensor_slices((np.array(raw_data['x']), np.array(raw_data['y'])))
        dataset = dataset.shuffle(50000, reshuffle_each_iteration=True)
        dataset = dataset.batch(self.batch_size, drop_remainder=False).prefetch(AUTOTUNE)

        return dataset
