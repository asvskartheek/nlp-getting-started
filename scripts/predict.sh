# train.sh {model_arch} {dataset}
allennlp predict --output-file experiments/predictions/$1.csv \
--use-dataset-reader \
--predictor simple_classifier \
--batch-size 64 \
--include-package src.dataset_reader \
--include-package src.model \
--include-package src.predict \
--silent \
experiments/results/$1/model.tar.gz data/$2.csv
