import argparse

DATASETS = ['femnist_iid', 'femnist_niid', 'shakespeare_niid']

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset',
                    help='name of dataset;',
                    type=str,
                    choices=DATASETS,
                    required=True)
    
    parser.add_argument('-model',
                    help='name of model;',
                    type=str,
                    required=True)
    
    parser.add_argument('--clients-per-round',
                    help='number of clients trained per round;',
                    type=int,
                    default=54)

    parser.add_argument('--num-rounds',
                    help='number of rounds to simulate;',
                    type=int,
                    default=40)
    
    parser.add_argument('--num-local-epochs',
                    help='number of epochs when clients train on data;',
                    type=int,
                    default=3)

    parser.add_argument('--GPU-id',
                    help='which gpu you use;',
                    type=int,
                    default=0)
    

    return parser.parse_args()