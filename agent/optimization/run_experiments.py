from subprocess import Popen




def run_reward_confirmation():

    # parser.add_argument('--gravity', type=float)
    # parser.add_argument('--distance', type=float)
    # parser.add_argument('--linear', type=float)
    # parser.add_argument('--angular', type=float)
    # parser.add_argument('--seed', type=float)

    seeds = [150, 215, 345, 556]
    experiments = [
        # {'angular': 7.6688, 'distance': 3.6034, 'gravity': 3.7649, 'linear': 19.7336},
        # {'angular': 9.3314, 'distance': 1.6425, 'gravity': 2.3464, 'linear': 16.3632},
        {'angular': 5.2441, 'distance': 3.6465, 'gravity': 2.1191, 'linear': 19.4849},
        {'angular': 10, 'distance': 5, 'gravity': 5, 'linear': 20}
    ]
    commands = []

    for i, exp in enumerate(experiments):
        for seed in seeds:
            commands.append(f'python reward_confirmation.py --gravity {exp["gravity"]} --distance {exp["distance"]} --linear {exp["linear"]} --angular {exp["angular"]} --seed {seed} --id {i+2}' )


    workers = 2
    for i in range(0, 500, workers):
        c = [commands[i+w] for w in range(workers)]
        processes = [Popen(cmd, shell=True) for cmd in c]
        for p in processes: p.wait()

    return


run_reward_confirmation()