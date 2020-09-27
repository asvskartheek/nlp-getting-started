import optuna

def objective(trial: optuna.Trial) -> float:
    trial.suggest_int("embedding_dim", 32, 256)

    executor = optuna.integration.allennlp.AllenNLPExecutor(
        trial=trial,  # trial object
        config_file="placeholder_config.jsonnet",  # jsonnet path
        serialization_dir=f"{trial.number}",  # directory for snapshots and logs
        metrics="best_validation_loss",
        include_package=['model', 'predict', 'dataset_reader']
    )
    return executor.run()

study = optuna.create_study(
    storage=None,  # save results in DB
    sampler=optuna.samplers.TPESampler(seed=24),
    study_name="finetune_boe_model",
    direction="minimize",
)

timeout = 60 * 60 * 10  # timeout (sec): 60*60*10 sec => 10 hours
study.optimize(
    objective,
    n_jobs=1,  # number of processes in parallel execution
    n_trials=30,  # number of trials to train a model
    timeout=timeout,  # threshold for executing time (sec)
)