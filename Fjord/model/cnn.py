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
        self.profile_num_samples = 0
        self.models_train_time = []
        self.model_list = []
        self.model = None
        self.create_models()
    
    def create_model(self, p):
        x = inputs = layers.Input((IMAGE_SIZE*IMAGE_SIZE*CHANNEL_SIZE))
        x = tf.keras.layers.Reshape((IMAGE_SIZE, IMAGE_SIZE, CHANNEL_SIZE))(x)
        x = tf.keras.layers.Conv2D(int(32*p), 5, padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = tf.keras.layers.Conv2D(int(64*p), 5, padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(int(1024*p), activation='relu')(x)
        x = tf.keras.layers.Dense(int(1024*p), activation='relu')(x)
        x = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)

        return models.Model(inputs, x, name='CNN')

    def create_models(self):
        num_tier = SYSTEM_PARAMS['NUM_TIER']
        for i in range(1, num_tier+1):
            model = self.create_model(i/num_tier)
            model.summary()
            model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr, momentum = 0.9, decay=1e-4),
                        loss = tf.keras.losses.SparseCategoricalCrossentropy())
            self.model_list.append(model)

        self.model = self.model_list[num_tier-1]

    def dropout_train(self, dataset_train, num_local_epochs, OD_index):
        
        self.model = self.model_list[OD_index]

        self.model.fit(dataset_train, epochs = num_local_epochs)

        return self.model.get_weights()

    def test(self, dataset_test):
        self.model = self.model_list[SYSTEM_PARAMS['NUM_TIER']-1]
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
        dataset = tf.data.Dataset.from_tensor_slices((np.array(raw_data['x']), np.array(raw_data['y'])))
        dataset = dataset.shuffle(50000, reshuffle_each_iteration=True)
        dataset = dataset.batch(self.batch_size, drop_remainder=False).prefetch(AUTOTUNE)

        return dataset
