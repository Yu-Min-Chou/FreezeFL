import numpy as np
import statistics

from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY, TIME_KEY, SYSTEM_PARAMS

class Server:
    
    def __init__(self, client_model, GPU_id, profile_result):
        self.client_model = client_model
        self.global_model = client_model.get_model_weight()
        self.pre_global_model = client_model.get_model_weight()
        self.profile_result = profile_result
        self.selected_clients = []
        self.updates = []
        self.fw = open('accu_' + str(GPU_id) + '.txt', 'a')
        self.sys_metrics = {BYTES_READ_KEY: 0,
                            BYTES_WRITTEN_KEY: 0,
                            LOCAL_COMPUTATIONS_KEY: 0,
                            TIME_KEY: 0}

    def select_clients(self, cur_round, possible_clients, num_clients=20):
        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(cur_round)
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)

        return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients]

    def train_model(self, num_local_epochs=1):
        clients = self.selected_clients

        count = 0

        for c in clients:
            c.model.set_model_weight(self.global_model)

            comp_time, read, written, time, num_samples, update, OD_index = c.train(num_local_epochs, self.profile_result)

            self.sys_metrics[BYTES_READ_KEY] += read
            self.sys_metrics[BYTES_WRITTEN_KEY] += written
            self.sys_metrics[LOCAL_COMPUTATIONS_KEY] += comp_time

            self.updates.append((time, num_samples, update, OD_index))

            count = count + 1

            if(count % 50 == 0):
                print("%d/%d clients has perform local training..." % (count, len(self.selected_clients)))

    def update_model(self):
        num_models = len(self.global_model)
        # Order Dropout Weight Aggregation
        total_samples = [0] * num_models
        base = [[[0] for i in range(len(self.updates[0][2]))] for i in range(num_models)]
        round_time = 0
        
        # Sum up different size models independently
        for (time, num_samples, update, OD_index) in self.updates:
            round_time = max(time, round_time)
            total_samples[OD_index] += num_samples
            tmp_list = [0] * len(self.updates[0][2])
            for i, v in enumerate(update):
                base[OD_index][i] += (num_samples * v.astype(np.float64))
        
        # Combine the model into the complete model
        if(total_samples[num_models-1] == 0):
            total_samples[num_models-1] = 1
            base[num_models-1] = self.global_model[num_models-1]

        complete_model = []
        for i in range(len(self.updates[0][2])):
            complete_model.append(np.zeros(np.shape(self.global_model[num_models-1][i])))

        for i in range(num_models):
            if total_samples[i] == 0: continue
            total_sample = np.sum(total_samples[i:])
            for j in range(len(self.updates[0][2])):
                shape_list = []
                tmp_shape = np.shape(base[i][j])
                complete_shape = np.shape(complete_model[j])
                for k in range(len(tmp_shape)):
                    shape_list.append((0, complete_shape[k]-tmp_shape[k]))
                shape_tuple = tuple(shape_list)

                partial_model = 0
                for k in range(i, num_models):
                    if total_samples[k] == 0: continue
                    partial_model += np.resize(base[k][j], np.shape(base[i][j]))

                total_sample = np.sum(total_samples[i:])
                padded_base = np.lib.pad(partial_model, shape_tuple, 'constant', constant_values=(0))
                complete_model[j] += padded_base / total_sample

                if(i == num_models - 1): continue
                partial_model = 0
                for k in range(i+1, num_models):
                    if total_samples[k] == 0: continue
                    partial_model += np.resize(base[k][j], np.shape(base[i][j]))
                
                total_sample = np.sum(total_samples[i+1:])
                padded_base = np.lib.pad(partial_model, shape_tuple, 'constant', constant_values=(0))
                complete_model[j] -= padded_base / total_sample

        for i in range(num_models):
            for j in range(len(self.updates[0][2])):
                tmp_shape = np.shape(self.global_model[i][j])
                self.global_model[i][j] = np.resize(complete_model[j], tmp_shape)

        self.sys_metrics[TIME_KEY] += round_time
        self.updates = []

    def update_utility(self):
        print("Todo")

    def update_selected_probablity(self, all_clients, cur_round):
        print("Todo")   

    def test_model(self, clients):
        losses = []
        accuracies = []

        for c in clients:
            c.model.set_model_weight(self.global_model)

            loss, accuracy = c.test()

            losses.append(loss)
            accuracies.append(accuracy)

        print("Test result: loss: %f, accuracy: %f" % (statistics.mean(losses), statistics.mean(accuracies)))
        self.fw.write("%f \n" % statistics.mean(accuracies))

    def get_clients_info(self, clients):
        if clients is None:
            clients = self.selected_clients

        ids = [c.id for c in clients]
        num_samples = {c.id: c.num_samples for c in clients}
        return ids, num_samples