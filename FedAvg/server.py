import numpy as np
import statistics

from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY, TIME_KEY

class Server:
    
    def __init__(self, client_model, GPU_id, profile_result):
        self.client_model = client_model
        self.global_model = client_model.get_model_weight()
        self.profile_result = profile_result
        self.selected_clients = []
        self.updates = []
        self.fw = open('accu_' + str(GPU_id) + '.txt', 'a')
        self.sys_metrics = {BYTES_READ_KEY: 0,
                            BYTES_WRITTEN_KEY: 0,
                            LOCAL_COMPUTATIONS_KEY: 0,
                            TIME_KEY: 0}
        self.proportion = []

    def select_clients(self, cur_round, possible_clients, num_clients=20):
        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(cur_round+45)
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)

        return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients]

    def train_model(self, num_local_epochs=1):
        clients = self.selected_clients

        count = 0

        for c in clients:
            c.model.set_model_weight(self.global_model)

            comp_time, read, written, time, num_samples, update = c.train(num_local_epochs, self.profile_result)

            self.sys_metrics[BYTES_READ_KEY] += read
            self.sys_metrics[BYTES_WRITTEN_KEY] += written
            self.sys_metrics[LOCAL_COMPUTATIONS_KEY] += comp_time

            self.updates.append((time, num_samples, update))

            count = count + 1

            self.proportion.append(comp_time/time)

            if(count % 50 == 0):
                print("%d/%d clients has perform local training..." % (count, len(self.selected_clients)))

    def update_model(self):
        total_weight = 0.
        base = [0] * len(self.updates[0][2])
        round_time = 0
        for (time, num_samples, update) in self.updates:
            round_time = max(round_time, time)
            total_weight += num_samples
            for i, v in enumerate(update):
                base[i] += (num_samples * v.astype(np.float32))

        weighted_avg = [v / total_weight for v in base]
        self.global_model = weighted_avg
        self.sys_metrics[TIME_KEY] += round_time
        
        self.updates = []

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
