MODEL_PARAMS = {
    'femnist_iid': (0.03, 64, 62), # lr, batch_size, num_classes
    'femnist_niid': (0.03, 64, 62), # lr, batch_size, num_classes
    'shakespeare_niid': (0.003, 256, 80, 80, 256), # lr, batch_size, seq_len, num_classes, num_hidden
}

SYSTEM_PARAMS = {
    # 'CAP_MIN': 0.5,
    # 'CAP_MEAN': 1,
    # 'CAP_SIGMA': 0.6,
    # 'BASIC_BANDWIDTH': 50,
    # Fjord
    'NUM_TIER': 4,

    # CNN
    'CAP_TIER': 8,
    'CAP_MIN': 0.5,
    'CAP_MAX': 3.0,
    'BASIC_COMPUTATION': 1.5,
    'BASIC_BANDWIDTH': 50

    # RNN
    # 'CAP_TIER': 8,
    # 'CAP_MIN': 0.5,
    # 'CAP_MAX': 3.0,
    # 'BASIC_COMPUTATION': 3,
    # 'BASIC_BANDWIDTH': 25
}

ACCURACY_KEY = 'accuracy'
BYTES_WRITTEN_KEY = 'bytes_written'
BYTES_READ_KEY = 'bytes_read'
LOCAL_COMPUTATIONS_KEY = 'local_computations'
NUM_ROUND_KEY = 'round_number'
NUM_SAMPLES_KEY = 'num_samples'
CLIENT_ID_KEY = 'client_id'
TIME_KEY = 'time'