import optuna





def view_all_results_ordered(trials, criteria=0):

    for trial in trials:
        if trial.value is None:
            trial.value = 0

    newlist = sorted(trials, reverse=True, key=lambda d: d.value) 
    for i in range(len(newlist)):
        if newlist[i].value > criteria:
            print(newlist[i].value, newlist[i].params)

    return


def analyse_reward_opt():

    storage = "sqlite:///reward_search.db"
    study = optuna.load_study(study_name='reward-search', storage=storage)
    trials = study.get_trials(deepcopy=True)
  


    view_all_results_ordered(trials, 1.6)

    
    return

def analyse_hp_opt():

    storage = "sqlite:///hp_opt_small.db"
    # study = optuna.load_study( study_name=None, storage=storage)
    studies = optuna.study.get_all_study_summaries(storage=storage)

    for study in studies:
        print(study.study_name, study.n_trials)
    # trials = study.get_trials(deepcopy=True)
  


    # view_all_results_ordered(trials, 1.6)

    
    return

# analyse_hp_opt()

analyse_reward_opt()