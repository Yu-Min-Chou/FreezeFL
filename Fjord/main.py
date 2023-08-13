import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils.args import parse_args
args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]= str(args.GPU_id)

import pickle
import importlib
import numpy as np

import random
import tensorflow as tf

from baseline_constants import MODEL_PARAMS, SYSTEM_PARAMS

from client import Client
from server import Server

from utils.model_utils import read_data

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
Fjord_path = 'profile_result'

def main():

    model_path = 'model.%s' % (args.model)
    print("model path: %s" % (model_path))
    
    mod = importlib.import_module(model_path)
    ClientModel = getattr(mod, 'ClientModel')

    num_rounds = args.num_rounds
    clients_per_round = args.clients_per_round

    fw = open(Fjord_path, 'rb')
    profile_result = pickle.load(fw)
    profile_result = profile_result[args.model]

    # Create 2 models
    model_params = MODEL_PARAMS[args.dataset]
    model_params_list = list(model_params)
    model_params = tuple(model_params_list)

    # Create client model, and share params with server model
    client_model = ClientModel(*model_params)

    # Create server
    server = Server(client_model, args.GPU_id, profile_result)

    # Create clients
    clients = setup_clients(args.dataset, client_model)
    client_ids, client_num_samples = server.get_clients_info(clients)
    print('Clients in Total: %d' % len(clients))

    # Simulate training
    for i in range(num_rounds):
        print('--- Round %d of %d: Training %d Clients ---' % (i + 1, num_rounds, clients_per_round))

        # Select clients to train this round
        server.select_clients(i, online(clients), num_clients=clients_per_round)
        c_ids, c_num_samples = server.get_clients_info(server.selected_clients)

        # Simulate server model training on selected clients' data
        server.train_model(num_local_epochs=args.num_local_epochs)

        # Update server model
        server.update_model()

        """The evaluation may need to update the utility and selected probability"""
        # # Update utility of clients
        # server.update_utility()

        # # Update select probability of clients 
        # server.update_selected_probablity(clients, i)

        # evaluation
        server.test_model(clients)
        
    for key in server.sys_metrics:
        server.fw.write("Key: %s, Value: %f\n" % (key, server.sys_metrics[key]))
    
    print("Training is finished")

def online(clients):
    """We assume all users are always online."""
    return clients


def create_clients(users, train_data, test_data, model):
    # mean, sigma = SYSTEM_PARAMS['CAP_MEAN'], SYSTEM_PARAMS['CAP_SIGMA']
    # random_capabilities = np.sort(np.random.normal(mean, sigma, len(users)))
    
    CAP_TIER, CAP_MAX, CAP_MIN = SYSTEM_PARAMS['CAP_TIER'], SYSTEM_PARAMS['CAP_MAX'], SYSTEM_PARAMS['CAP_MIN']
    capability = []
    for i in range(CAP_TIER):
        ability = ((CAP_MAX - CAP_MIN) / (CAP_TIER-1)) * i + CAP_MIN
        tmp_list = [ability] * int(len(users) / CAP_TIER)
        capability += tmp_list
    if len(capability) < len(users): capability += ([CAP_MAX] * (len(users) - len(capability)))

    deafult_selected_probability = 1 / len(users)
    default_utitlity = 10
    clients = [Client(u, train_data[u], test_data[u], model, capability[i], deafult_selected_probability, default_utitlity, i/len(users)) for i, u in enumerate(users)]
    return clients


def setup_clients(dataset, model=None):
    """Instantiates clients based on given train and test data directories.

    Return:
        all_clients: list of Client objects.
    """
    train_data_dir = os.path.join('/home/potter0694/Freezing/FedAvg/data', dataset, 'train')
    test_data_dir = os.path.join('/home/potter0694/Freezing/FedAvg/data', dataset, 'test')

    users, train_data, test_data = read_data(train_data_dir, test_data_dir)

    clients = create_clients(users, train_data, test_data, model)

    return clients


if __name__ == '__main__':
    main()
