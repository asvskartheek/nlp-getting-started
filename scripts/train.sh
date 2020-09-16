# train.sh {model_arch}
allennlp train \
	experiments/configs/$1.jsonnet \
	-s experiments/results/$1 \
	--include-package src.dataset_reader \
	--include-package src.model