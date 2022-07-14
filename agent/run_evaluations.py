from subprocess import Popen



def eval_test():
    """
    runs a batch of experiments for the core work for this dissertation
    """
    reward_params = {'angular': 5.2441, 'distance': 3.6465, 'gravity': 2.1191, 'linear': 19.4849}
    experiments = [
        {'phase':True, 'contacts':True, 'targets':False, 'vision':False},
    ]

    commands = []

    for i, exp in enumerate(experiments):
        
        commands.append(f'python train.py \
            --save_dir {"train_state_models/0_150"} \
                \
            --gravity {reward_params["gravity"]} \
            --distance {reward_params["distance"]}\
            --linear {reward_params["linear"]} \
            --angular {reward_params["angular"]} \
                \
            --use_contacts {exp["contacts"]} \
            --use_targets {exp["targets"]} \
            --use_phase {exp["phase"]} \
            --use_vision {exp["vision"]} \
                \
            --env {"eval_9"} \
            --eval \
            ')


    workers = 1
    for i in range(0, 500, workers):
        try:
            c = [commands[i+w] for w in range(workers)]
            processes = [Popen(cmd, shell=True) for cmd in c]
            for p in processes: p.wait()
        except:
            break

    return


eval_test()