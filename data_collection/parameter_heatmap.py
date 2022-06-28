from copy import deepcopy
import optuna

if __name__ == '__main__':

    storage = "sqlite:///demo_step_18_f_28_32.db"
    # study = optuna.create_study(study_name='step_012_only', storage=storage, directions=["maximize"])

    #step_height=0.12 -> step_014_only
    #step_height=0.14 -> step_014_final2
    #step_height=0.16 -> step_016_only
    #step_height=0.18 -> step_018_only
    #step_height=0.20 -> step_020_only
    #step_height=0.22 -> step_022_only
    #step_height=0.24 -> step_024_only

    study = optuna.load_study(study_name='step_020_only', storage=storage)
    # study.optimize(_run_example, n_trials=1000, gc_after_trial=True, show_progress_bar=True )
    trials = study.get_trials(deepcopy=True)
    for trial in trials:
        if trial.value is None:
            trial.value = 0


    # newlist = sorted(trials, reverse=True, key=lambda d: d.value) 
    # for i in range(200):
    #     print(newlist[i].value, newlist[i].params)

    fig = optuna.visualization.plot_contour(study, params=["frequency", "duty_factor"])
    fig.show()