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
            "type": "bag_of_embeddings",
            "embedding_dim": 100,
			"averaged": true
        }
    },
    "data_loader": {
        "batch_size": 8,
        "shuffle": true
    },
    "trainer": {
        "optimizer": "adam",
        "num_epochs": 5
    }
}