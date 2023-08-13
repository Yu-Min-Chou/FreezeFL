import random
import warnings
import time
import tensorflow as tf

from baseline_constants import SYSTEM_PARAMS

class Client:
    
    def __init__(self, client_id, train_data={'x' : [],'y' : []}, test_data={'x' : [],'y' : []}, model=None, ability = 1, selected_probability = 0, utility = 1):
        self._model = model
        self.id = client_id
        self.train_data = train_data
        self.test_data = test_data
        self.train_dataset = self.model.process_data(self.train_data)
        self.test_dataset = self.model.process_data(self.test_data)
        self.participate_time = 0
        self.wu_participate_time = 0
        self.selected_probability = selected_probability
        self.utility = utility
        self.tier_id = ability[0]

        self.comp_ability = SYSTEM_PARAMS['BASIC_COMPUTATION'] * max(SYSTEM_PARAMS['CAP_MIN'], ability[1])
        self.comm_ability = SYSTEM_PARAMS['BASIC_BANDWIDTH'] * max(SYSTEM_PARAMS['CAP_MIN'], ability[1])

    def train(self, num_local_epochs, soft_deadline, profile_result):
        self.participate_time += 1
        self.wu_participate_time += 1

        update, num_frozen_layers = self.model.freeze_train(self.train_dataset, num_local_epochs, soft_deadline, self.comp_ability, self.comm_ability, self.num_train_samples, profile_result)

        read = self.model.get_model_size(full = True)
        written = self.model.get_model_size(full = False)
        num_samples = self.num_train_samples

        comm_time = (read + written) / self.comm_ability
        comp_time = profile_result[num_frozen_layers] * num_samples * num_local_epochs / self.comp_ability
        total_time = comm_time + comp_time

        return comp_time, read, written, total_time, num_samples, update, num_frozen_layers

    def test(self):
        loss, accuracy = self.model.test(self.test_dataset)
        return loss, accuracy

    def task_time(self, profile_result, num_local_epochs):
        
        model_size = self.model.get_model_size(full = True)
        comm_time = model_size*2 / self.comm_ability
        comp_time = profile_result[0] * self.num_train_samples * num_local_epochs / self.comp_ability

        return comm_time + comp_time

    @property
    def num_test_samples(self):
        """Number of test samples for this client.

        Return:
            int: Number of test samples for this client
        """
        if self.test_data is None:
            return 0
        return len(self.test_data['y'])

    @property
    def num_train_samples(self):
        """Number of train samples for this client.

        Return:
            int: Number of train samples for this client
        """
        if self.train_data is None:
            return 0
        return len(self.train_data['y'])

    @property
    def num_samples(self):
        """Number samples for this client.

        Return:
            int: Number of samples for this client
        """
        train_size = 0
        if self.train_data is not None:
            train_size = len(self.train_data['y'])

        test_size = 0
        if self.test_data is not  None:
            test_size = len(self.test_data['y'])
        return train_size + test_size

    @property
    def model(self):
        """Returns this client reference to model being trained"""
        return self._model

    @model.setter
    def model(self, model):
        warnings.warn('The current implementation shares the model among all clients.'
                      'Setting it on one client will effectively modify all clients.')
        self._model = model
