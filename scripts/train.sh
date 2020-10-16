# train.sh {model_arch}
wandb_allennlp \
	--subcommand=train \
	--config_file=experiments/configs/bert.jsonnet \
	--include-package=src.models.simple_model \
	--include-package=src.dataset_reader