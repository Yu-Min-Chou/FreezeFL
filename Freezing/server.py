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
        self.soft_deadline = SYSTEM_PARAMS['SOFT_DEADLINE']
        self.sys_metrics = {BYTES_READ_KEY: 0,
                            BYTES_WRITTEN_KEY: 0,
                            LOCAL_COMPUTATIONS_KEY: 0,
                            TIME_KEY: 0}

        self.trainable_index = []
        self.trainable_record = self.client_model.trainable_record
        for i in range(len(self.trainable_record)):
            if(self.trainable_record[i] > 0):
                self.trainable_index.append(i)

        print(self.profile_result)

    def select_clients(self, cur_round, possible_clients, num_clients=20):
        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(cur_round)
        if(SYSTEM_PARAMS['CLIENT_SELECTION']):
            self.selected_clients = np.random.choice(possible_clients, num_clients, p = [c.selected_probability for c in possible_clients], replace=False)
        else:
            self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)

        return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients]

    def train_model(self, num_local_epochs=1):
        clients = self.selected_clients
        count = 0

        for c in clients:
            c.model.set_model_weight(self.global_model)

            comp_time, read, written, time, num_samples, update, num_frozen_layers = c.train(num_local_epochs, self.soft_deadline, self.profile_result)

            self.sys_metrics[BYTES_READ_KEY] += read
            self.sys_metrics[BYTES_WRITTEN_KEY] += written
            self.sys_metrics[LOCAL_COMPUTATIONS_KEY] += comp_time

            self.updates.append((time, num_samples, update, num_frozen_layers))

            count = count + 1

            if(count % 50 == 0):
                print("%d/%d clients has perform local training..." % (count, len(self.selected_clients)))

    def update_model(self):
        deadline_factor = SYSTEM_PARAMS['DEADLINE_FACTOR']

        # Layerwise Weight Aggregation
        if(SYSTEM_PARAMS['LAYER_AGGREGATION']):
            total_samples = [0.] * len(self.updates[0][2])
            base = [0] * len(self.updates[0][2])
            average_time = []
            weighted_avg = []
            round_time = 0

            for (time, num_samples, update, num_frozen_layers) in self.updates:
                average_time.append(time)
                round_time = max(round_time, time)
                for i in range(len(self.trainable_index)):
                    if(num_frozen_layers <= self.trainable_index[i]):
                        layer_index = self.trainable_index[i]
                        cur_index = sum(self.trainable_record[:layer_index])
                        for j in range(cur_index, cur_index+self.trainable_record[layer_index]):
                            base[j] += num_samples * update[j].astype(np.float32)
                            total_samples[j] += num_samples
            
            for i, v in enumerate(zip(base, total_samples)):
                if(v[1] == 0):
                    weighted_avg.append(self.global_model[i])
                else:
                    weighted_avg.append(v[0] / v[1])
        # Weighted Aggregation
        else:
            total_weight = 0.
            base = [0] * len(self.updates[0][2])
            average_time = []
            round_time = 0
            for (time, num_samples, update, num_frozen_layers) in self.updates:
                average_time.append(time)
                round_time = max(round_time, time)
                total_weight += num_samples
                for i, v in enumerate(update):
                    base[i] += (num_samples * v.astype(np.float32))

            weighted_avg = [v / total_weight for v in base]

        self.global_model = weighted_avg
        self.sys_metrics[TIME_KEY] += round_time
        self.soft_deadline = (1-deadline_factor) * self.soft_deadline + deadline_factor * statistics.mean(average_time)
        print("soft_deadline: %f" % self.soft_deadline)
        
        # self.updates = []

    def update_utility(self):
        utility_factor = SYSTEM_PARAMS['UTILITY_FACTOR']
        clients = self.selected_clients
        global_model_diff = np.asarray(self.global_model, dtype=object) - np.asarray(self.pre_global_model, dtype=object)

        for i, client in enumerate(clients):
            update = self.updates[i][2]
            update_diff = np.asarray(update, dtype=object) - np.asarray(self.pre_global_model, dtype=object)
            utility = 0
            for a, b in zip(global_model_diff, update_diff):
                a = a.reshape(-1)
                b = b.reshape(-1)
                utility += np.sum(np.inner(a, b)) / a.shape[0]

            utility = (1-utility_factor) * client.utility + utility_factor * utility
            client.utility = max(1e-05, utility) * (len(self.trainable_record) - self.updates[i][3])

        self.pre_global_model = self.global_model

        self.updates = []

    def update_selected_probablity(self, all_clients, cur_round):
        sum_utility = 0
        warmup_period = SYSTEM_PARAMS['WARMUP_PERIOD']

        for client in all_clients:
            sum_utility += client.utility

        for client in all_clients:
            client.selected_probability = client.utility / sum_utility

        # warmup mechanism
        if(SYSTEM_PARAMS['WARMUP']):
            if(cur_round % warmup_period == 0):
                mean_utility = sum_utility / len(all_clients)
                for client in all_clients:
                    if(client.wu_participate_time == 0):
                        client.wu_participate_time = 1
                    if(client.utility > mean_utility):
                        client.utility -= np.sqrt(2 * np.log(warmup_period) / client.wu_participate_time)
                        client.utility = max(mean_utility, client.utility)
                    else:
                        client.utility += np.sqrt(2 * np.log(warmup_period) / client.wu_participate_time)
                        client.utility = min(mean_utility, client.utility)
                    client.wu_participate_time = 0
            

    def test_model(self, clients):
        losses = []
        accuracies = []

        for c in clients:
            c.model.set_model_weight(self.global_model)

            loss, accuracy = c.test()

            losses.append(loss)
            accuracies.append(accuracy)

        print("Test result: loss: %f, accuracy: %f" % (statistics.mean(losses), statistics.mean(accuracies)))
        self.fw.write("%f\n" % (statistics.mean(accuracies)))

    def get_clients_info(self, clients):
        if clients is None:
            clients = self.selected_clients

        ids = [c.id for c in clients]
        num_samples = {c.id: c.num_samples for c in clients}
        return ids, num_samples