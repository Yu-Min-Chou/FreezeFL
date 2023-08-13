MODEL_PARAMS = {
    'femnist_iid': (0.03, 64, 62), # lr, batch_size, num_classes
    'femnist_niid': (0.03, 64, 62), # lr, batch_size, num_classes
    'shakespeare_niid': (0.003, 256, 80, 80, 256), # lr, batch_size, seq_len, num_classes, num_hidden
}

SYSTEM_PARAMS = {
    # Layerwise params
    'SOFT_DEADLINE': 1,
    'PERF_FACTOR': 4,
    'DEADLINE_FACTOR': 0.4,
    'UTILITY_FACTOR': 0.9,
    'CLIENT_SELECTION': True,
    'WARMUP': True,
    'WARMUP_PERIOD': 60,
    'LAYER_AGGREGATION': True,

    # CNN
    'CAP_TIER': 8,
    'CAP_MIN': 0.5,
    'CAP_MAX': 3.0,
    'BASIC_COMPUTATION': 0.4,
    'BASIC_BANDWIDTH': 100

    # RNN
    # 'CAP_TIER': 8,
    # 'CAP_MIN': 0.5,
    # 'CAP_MAX': 3.0,
    # 'BASIC_COMPUTATION': 3,
    # 'BASIC_BANDWIDTH': 25
}

BYTES_WRITTEN_KEY = 'bytes_written'
BYTES_READ_KEY = 'bytes_read'
LOCAL_COMPUTATIONS_KEY = 'local_computations'
TIME_KEY = 'time'