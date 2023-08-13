import sys
import random
import numpy as np
import statistics

from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY, TIME_KEY, SYSTEM_PARAMS

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

    def select_clients(self, cur_round, possible_clients, num_clients, num_local_epochs):
        # Plus the v_queue of all clients
        for c in possible_clients:
            c.v_queue += (1/len(possible_clients) * num_clients)

        # Sort the clients based on the "age"
        sorted_clients = sorted(possible_clients, key=lambda x: x.v_queue, reverse=True)

        # choose the most powerful client of a group
        fair_factor = SYSTEM_PARAMS['FAIR_FACTOR']
        fair_score = sys.float_info.max
        for max_client in possible_clients:
            client_pool = []
            client_pool.append(possible_clients[54+cur_round])
            sum_v_queue = 0
            max_time = max_client.task_time(self.profile_result, num_local_epochs)
            for c in sorted_clients:
                if(c.task_time(self.profile_result, num_local_epochs) < max_time):
                    client_pool.append(c)
                    sum_v_queue += c.v_queue
                if(len(client_pool) == num_clients):
                    # calculate the score of this group
                    score = fair_factor * max_time - sum_v_queue
                    # print(max_time, sum_v_queue, score)
                    if(score < fair_score):
                        self.selected_clients = client_pool
                        fair_score = score
                    break

        # Reduce the v_queue of selected clients by one
        for c in self.selected_clients:
            c.v_queue = max(0, c.v_queue-1)

        return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients]

    def train_model(self, num_local_epochs=1):
        clients = self.selected_clients

        count = 0

        for c in clients:
            # print("Clinet %s performs local training, with %d samples" % (c.id, c.num_train_samples))
            c.model.set_model_weight(self.global_model)

            comp_time, read, written, time, num_samples, update = c.train(num_local_epochs, self.profile_result)

            self.sys_metrics[BYTES_READ_KEY] += read
            self.sys_metrics[BYTES_WRITTEN_KEY] += written
            self.sys_metrics[LOCAL_COMPUTATIONS_KEY] += comp_time

            self.updates.append((time, num_samples, update))

            count = count + 1

            if(count % 50 == 0):
                print("%d/%d clients has perform local training..." % (count, len(self.selected_clients)))

    def update_model(self):
        # Weighted Aggregation
        total_samples = 0.
        base = [0] * len(self.updates[0][2])
        round_time = 0
        for (time, num_samples, update) in self.updates:
            round_time = max(round_time, time)
            total_samples += num_samples
            for i, v in enumerate(update):
                base[i] += (num_samples * v.astype(np.float64))

        weighted_avg = [v / total_samples for v in base]

        self.sys_metrics[TIME_KEY] += round_time
        self.global_model = weighted_avg
        
        self.updates = []

    # def update_selected_probablity(self, all_clients, cur_round):
    #     print("Todo")    

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