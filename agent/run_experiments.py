from subprocess import Popen


def test():
    """
    runs a batch of experiments for the core work for this dissertation
    """
    seeds = [150, 215, 345, 556]
    reward_params = {'angular': 5.2441, 'distance': 3.6465, 'gravity': 2.1191, 'linear': 19.4849}
    hyper_params = {'lr':0.0005, 'lr_decay':1e-7, 'entropy_coef':0.0000026752, 'batch_size':500, 'roll_length':20000, 'num_epochs':5}

    experiments = [
        {'phase':False, 'contacts':False, 'targets':False, 'vision':False},
        {'phase':True, 'contacts':False, 'targets':False, 'vision':False},
        {'phase':True, 'contacts':True, 'targets':False, 'vision':False},
        {'phase':False, 'contacts':True, 'targets':True, 'vision':False},
        {'phase':False, 'contacts':True, 'targets':True, 'vision':True},
        {'phase':True, 'contacts':True, 'targets':True, 'vision':True}
    ]

    commands = []

    for i, exp in enumerate(experiments):
        for seed in seeds:
            commands.append(f'python train.py \
                --id {i} \
                --seed {seed} \
                --log_dir {"test_logs"} \
                --save_dir {"test_models"} \
                    \
                --gravity {reward_params["gravity"]} \
                --distance {reward_params["distance"]}\
                --linear {reward_params["linear"]} \
                --angular {reward_params["angular"]} \
                    \
                --lr {hyper_params["lr"]} \
                --lr_decay {hyper_params["lr_decay"]} \
                --entropy_coef {hyper_params["entropy_coef"]} \
                --batch_size {hyper_params["batch_size"]} \
                --rollout_length {hyper_params["roll_length"]} \
                --num_epochs {hyper_params["num_epochs"]} \
                    \
                --use_contacts {exp["contacts"]} \
                --use_targets {exp["targets"]} \
                --use_phase {exp["phase"]} \
                --use_vision {exp["vision"]} \
                    \
                --max_timesteps {int(20e3)} \
                    \
                --env {"train"} \
                --train \
                ')


    workers = 2
    for i in range(0, 500, workers):
        c = [commands[i+w] for w in range(workers)]
        processes = [Popen(cmd, shell=True) for cmd in c]
        for p in processes: p.wait()

    return


def training_experiments_stage1():
    """
    runs a batch of experiments for the core work for this dissertation
    """
    seeds = [150, 215, 345, 556]
    reward_params = {'angular': 5.2441, 'distance': 3.6465, 'gravity': 2.1191, 'linear': 19.4849}
    hyper_params = {'lr':0.0005, 'lr_decay':1e-7, 'entropy_coef':0.0000026752, 'batch_size':500, 'roll_length':20000, 'num_epochs':5}

    experiments = [
        {'phase':True, 'contacts':True, 'targets':False, 'vision':False},
        # {'phase':False, 'contacts':False, 'targets':False, 'vision':False},
        # {'phase':True, 'contacts':False, 'targets':False, 'vision':False},
        # {'phase':False, 'contacts':True, 'targets':True, 'vision':False},
        # {'phase':False, 'contacts':True, 'targets':True, 'vision':True},
        # {'phase':True, 'contacts':True, 'targets':True, 'vision':True}
    ]

    commands = []

    for i, exp in enumerate(experiments):
        for seed in seeds:
            commands.append(f'python train.py \
                --id {i} \
                --seed {seed} \
                --log_dir {"train_state_final"} \
                --save_dir {"train_state_models"} \
                    \
                --gravity {reward_params["gravity"]} \
                --distance {reward_params["distance"]}\
                --linear {reward_params["linear"]} \
                --angular {reward_params["angular"]} \
                    \
                --lr {hyper_params["lr"]} \
                --lr_decay {hyper_params["lr_decay"]} \
                --entropy_coef {hyper_params["entropy_coef"]} \
                --batch_size {hyper_params["batch_size"]} \
                --rollout_length {hyper_params["roll_length"]} \
                --num_epochs {hyper_params["num_epochs"]} \
                    \
                --use_contacts {exp["contacts"]} \
                --use_targets {exp["targets"]} \
                --use_phase {exp["phase"]} \
                --use_vision {exp["vision"]} \
                    \
                --max_timesteps {int(10e6)} \
                    \
                --env {"train"} \
                --train \
                ')


    workers = 2
    for i in range(0, 500, workers):
        try:
            c = [commands[i+w] for w in range(workers)]
            processes = [Popen(cmd, shell=True) for cmd in c]
            for p in processes: p.wait()
        except:
            break

    return

def reward_features_exp():
    """
    runs random search for reward parameters

    - {'gravity':True, 'lin':True, 'ang':True, 'vel_verison':0, 'dist':True, 'fall':True, 'target':True}
    """

    experiments = [
        {'gravity':True, 'lin':True, 'ang':True, 'vel_verison':0, 'dist':False, 'fall':False, 'target':False},
        {'gravity':True, 'lin':True, 'ang':True, 'vel_verison':1, 'dist':False, 'fall':False, 'target':False},
        {'gravity':True, 'lin':True, 'ang':True, 'vel_verison':2, 'dist':False, 'fall':False, 'target':False},

        {'gravity':True, 'lin':True, 'ang':True, 'vel_verison':0, 'dist':True, 'fall':False, 'target':False},
        {'gravity':True, 'lin':True, 'ang':True, 'vel_verison':1, 'dist':True, 'fall':False, 'target':False},
        {'gravity':True, 'lin':True, 'ang':True, 'vel_verison':2, 'dist':True, 'fall':False, 'target':False},

        {'gravity':True, 'lin':True, 'ang':True, 'vel_verison':0, 'dist':True, 'fall':True, 'target':False},
        {'gravity':True, 'lin':True, 'ang':True, 'vel_verison':1, 'dist':True, 'fall':True, 'target':False},
        {'gravity':True, 'lin':True, 'ang':True, 'vel_verison':2, 'dist':True, 'fall':True, 'target':False},

        {'gravity':True, 'lin':True, 'ang':True, 'vel_verison':0, 'dist':True, 'fall':False, 'target':True},
        {'gravity':True, 'lin':True, 'ang':True, 'vel_verison':1, 'dist':True, 'fall':False, 'target':True},
        {'gravity':True, 'lin':True, 'ang':True, 'vel_verison':2, 'dist':True, 'fall':False, 'target':True},

        {'gravity':True, 'lin':True, 'ang':True, 'vel_verison':0, 'dist':True, 'fall':True, 'target':True},
        {'gravity':True, 'lin':True, 'ang':True, 'vel_verison':1, 'dist':True, 'fall':True, 'target':True},
        {'gravity':True, 'lin':True, 'ang':True, 'vel_verison':2, 'dist':True, 'fall':True, 'target':True},
        
    ]


    return

def reward_parameter_search():
    """
    runs random search for reward parameters
    """


    commands = []

    for i in range(2):
        
        commands.append(f'python ./optimization/reward_optimization.py \
            --log_dir {"reward_search_long_logs"} \
            --save_dir {"reward_serch_long_models"} \
            ')


    workers = 2
    for i in range(0, 500, workers):
        try:
            c = [commands[i+w] for w in range(workers)]
            processes = [Popen(cmd, shell=True) for cmd in c]
            for p in processes: p.wait()
        except:
            break

    return


def hyper_parameter_search():
    """
    runs random search for reward parameters
    """


    commands = []

    for i in range(2):
        
        commands.append(f'python ./optimization/hp_optimization.py \
            --log_dir {"hp_search_long_logs"} \
            --save_dir {"hp_serch_long_models"} \
            ')


    workers = 2
    for i in range(0, 500, workers):
        try:
            c = [commands[i+w] for w in range(workers)]
            processes = [Popen(cmd, shell=True) for cmd in c]
            for p in processes: p.wait()
        except:
            break

    return

def hp_confirmation():

    seeds = [150, 215, 345, 556]
    experiments = [
        {'lr':0.001, 'lr_decay':1e-7, 'entropy_coef':0.000005, 'batch_size':4000, 'roll_length':20000, 'num_epochs':10, 'network':'small'},
        {'lr':0.001, 'lr_decay':1e-7, 'entropy_coef':0.000005, 'batch_size':500, 'roll_length':20000, 'num_epochs':20, 'network':'medium'},
        {'lr':0.003, 'lr_decay':1e-7, 'entropy_coef':0.000297, 'batch_size':500, 'roll_length':20000, 'num_epochs':20, 'network':'small'},
        {'lr':0.003, 'lr_decay':1e-7, 'entropy_coef':0.000964, 'batch_size':500, 'roll_length':20000, 'num_epochs':20, 'network':'large'},
        {'lr':0.001, 'lr_decay':1e-7, 'entropy_coef':0.000008, 'batch_size':500, 'roll_length':20000, 'num_epochs':10, 'network':'large'}
    ]
    commands = []

    for i, exp in enumerate(experiments):
        for seed in seeds:
            commands.append(f'python ./optimization/hp_confirmation.py --lr {exp["lr"]} --lr_decay {exp["lr_decay"]} --entropy_coef {exp["entropy_coef"]} --batch_size {exp["batch_size"]} \
                --roll_length {exp["roll_length"]} --num_epochs {exp["num_epochs"]} --network {exp["network"]}  --seed {seed} --id {i+2}' )


    workers = 2
    for i in range(0, 500, workers):
        c = [commands[i+w] for w in range(workers)]
        processes = [Popen(cmd, shell=True) for cmd in c]
        for p in processes: p.wait()

    return

  
# test()
# training_experiments_stage1()
# reward_parameter_search()
# hyper_parameter_search()
hp_confirmation()