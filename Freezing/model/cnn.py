import numpy as np
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
        self.trainable_record = []

        for layer in self.model.layers:
            self.trainable_record.append(len(layer.weights))
    
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

    def freeze_train(self, dataset_train, num_local_epochs, soft_deadline, comp_ability, comm_ability, num_samples, profile_result):
        layers_utility = []
        original_model_weight = self.get_model_weight()

        del self.model
        self.model = self.create_model()
        self.set_model_weight(original_model_weight)

        # Run a complete epoch to collect needed information
        for layer in self.model.layers:
            layer.trainable = True

        self.model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr, momentum = 0.9, decay=1e-4),
                        loss = tf.keras.losses.SparseCategoricalCrossentropy())

        self.model.fit(dataset_train, epochs = 1)

        tmp_model_weight = self.get_model_weight()

        weight_diff = (np.asarray(tmp_model_weight, dtype=object) - np.asarray(original_model_weight, dtype=object)).tolist()
        self.set_model_weight(weight_diff)

        # calculate layer utility
        for layer in self.model.layers:
            square_sum = 0
            num_neuron = 0
            for weights in layer.weights:
                weights_np = np.absolute(weights.numpy()).reshape(-1)
                square_sum += np.sum(weights_np)
                num_neuron += weights_np.shape[0]

            # print(square_sum, num_neuron)
            if num_neuron == 0:
                layers_utility.append(0)
            else:
                layers_utility.append(np.sqrt(square_sum / num_neuron))

        # print(layers_utility)

        # calculate how many layers should be freeze
        max_utility = -1
        num_frozen_layers = 0
        # print(comp_ability, comm_ability)
        for i, utility in enumerate(layers_utility):
            estimate_comp_time = profile_result[i] * num_samples * num_local_epochs / comp_ability
            estimate_comm_time = (self.get_model_size(full = True) + self.get_model_size(full = False)) / comm_ability
            estimate_time = estimate_comp_time + estimate_comm_time
            # print("estimate_time: %f" % estimate_time)
            time_penalty = (1 if soft_deadline < estimate_time else 0) * SYSTEM_PARAMS['PERF_FACTOR']
            total_utility = np.sum(layers_utility[i:]) * (soft_deadline / estimate_time)**(time_penalty)
            # print((soft_deadline / estimate_time)**(time_penalty))
            # print(total_utility)
            # print("Time penalty: %d, total_utility: %f" % (time_penalty, total_utility))
            if total_utility > max_utility:
                num_frozen_layers = i
                max_utility = total_utility

            self.model.layers[i].trainable = False

        # print(max_utility)
        # print(num_frozen_layers)
        for i, layer in enumerate(self.model.layers):
            if(i < num_frozen_layers):
                layer.trainable = False
            else:
                layer.trainable = True

        self.set_model_weight(original_model_weight)

        self.model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr, momentum = 0.9, decay=1e-4),
                        loss = tf.keras.losses.SparseCategoricalCrossentropy())

        # perform remain local training

        self.model.fit(dataset_train, epochs = num_local_epochs)

        trained_model_weight = self.get_model_weight()

        # Weird tensorflow memory leakage???

        return trained_model_weight, num_frozen_layers

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

    def get_model_size(self, full=True):
        trainableParams = np.sum([np.prod(v.get_shape()) for v in self.model.trainable_weights])
        nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in self.model.non_trainable_weights])
        # unit: MB

        if(full):
            return (trainableParams + nonTrainableParams) * (32 / 8) / 1024**2
        else:
            return (trainableParams) * (32 / 8) / 1024**2

    def process_data(self, raw_data):
        dataset = tf.data.Dataset.from_tensor_slices((np.array(raw_data['x']), np.array(raw_data['y'])))
        dataset = dataset.shuffle(50000, reshuffle_each_iteration=True)
        dataset = dataset.batch(self.batch_size, drop_remainder=False).prefetch(AUTOTUNE)

        return dataset
