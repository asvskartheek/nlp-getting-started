local batch_size = 8;
local num_epochs = 5;
local seed = 0;
local cuda_device = 0;

local embedding_dim = 144;

{
  numpy_seed: seed,
  pytorch_seed: seed,
  random_seed: seed,
  dataset_reader: {
    lazy: false,
    type: 'classification-csv',
    tokenizer: {
      type: 'spacy',
    },
    token_indexers: {
      tokens: {
        type: 'single_id',
        lowercase_tokens: true,
      },
    },
  },
  datasets_for_vocab_creation: ['train'],
  train_data_path: 'data/train.csv',
  validation_data_path: 'data/valid.csv',
  model: {
    type: 'simple_classifier',
    embedder: {
      token_embedders: {
        tokens: {
          embedding_dim: embedding_dim,
        },
      },
    },
    encoder: {
      "type": "bag_of_embeddings",
	  "embedding_dim": embedding_dim,
	  "averaged": true
    }
  },
  data_loader: {
    shuffle: true,
    batch_size: batch_size,
  },
  trainer: {
    num_epochs: num_epochs,
    optimizer: "adam",
    validation_metric: '+accuracy',
  },
}