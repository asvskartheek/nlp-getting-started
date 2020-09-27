{
    numpy_seed: 0,
    pytorch_seed: 0,
    random_seed: 0,
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
        "type": "attention_classifier",
        "embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 50
                }
            }
        },
        "encoder": {
            "type": "bag_of_embeddings",
            "embedding_dim": 50,
			"averaged": true
		},
		"attention": {
			"type": "additive",
			"vector_dim": 50,
            "matrix_dim": 50,
			"normalize": true
		}
    },
    "data_loader": {
        "batch_size": 8,
        "shuffle": true
    },
    "trainer": {
        "cuda_device": 0,
        "optimizer": "adam",
        "num_epochs": 5
    }
}