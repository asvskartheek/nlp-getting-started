{
	"dataset_reader" : {
		"type": "classification-csv",
		"token_indexers": {
			"tokens": {
				"type": "single_id"
			}
		}
	},
	"train_data_path": "data/train.csv",
	"validation_data_path": "data/valid.csv",
	"model": {
		"type": "simple_classifier",
		"embedder": {
			"token_embedders": {
				"tokens": {
					"type": "embedding",
					"embedding_dim": 100
				}
			}
		},
		"encoder": {
			"type": "cnn",
			"embedding_dim": 100,
			"num_filters": 24,
			"output_dim": 100
		}
	},
	"data_loader": {
		"batch_size": 32,
		"shuffle": true
	},
	"trainer": {
		"optimizer": "adam",
		"num_epochs": 5
	}
}