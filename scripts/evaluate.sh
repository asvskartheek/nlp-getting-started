# train.sh {model_arch} {dataset}
allennlp evaluate \
	experiments/results/$1 \
	data/$2.csv \
	--include-package src.dataset_reader \
	--include-package src.model