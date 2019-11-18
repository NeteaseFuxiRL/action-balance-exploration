

DEFAULT_PARAMS = {
    # env
    'regular_factor': None,
    'lr': 0.1,  # learning rate
    'lr_decay_rate': 0.99,
    'lr_decay_steps': 1e5,
    'max_grad_norm': 10,
    # residual
    'residual_layer_num': 6,  # number of layers in the critic/actor networks
    'output_channel': 256,
    'kernel_size': (3, 3),
    'use_batch_normalize': True,
    # value head
    'value_hidden': 256,  # number of neurons in each hidden layers
    'value_conv_kernel': (1, 1),
    'value_output_channel': 1,
    # policy head
    'policy_conv_kernel': (1, 1),
    'policy_output_channel': 2,
    # train
    'batch_size': 20,  # 1024
    'buffer_size': int(100000),  # 2e5, 1e6
    # 'batch_size': 128,
    # 'buffer_size': int(1e5),
    'replay_buffer_type': 'dict',  # 'double', 'dict'
    'buffer_reset_freq': 1000,
    'c_puct': 10,
    'only_positive': False,
    'prioritized_replay': True,
    'prioritized_replay_alpha': 0.6,
    'prioritized_replay_beta': 0.8,
    'prioritized_replay_beta_iters': 1000000,
    'total_times': float('inf'),  # float('inf')
    'train_freq': 5,
    'train_times': 100,
    'print_freq': 10,
    'save_freq': int(10e6) - 1,
    'state_one_hot_class_num': None,
}
