import tempfile
from typing import Dict, Iterable, List, Tuple

import torch

import allennlp
from allennlp.common import JsonDict
from allennlp.data import DataLoader, DatasetReader, Instance
from allennlp.data import Vocabulary
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from allennlp.models import Model
from allennlp.predictors import Predictor
import numpy as np

@Predictor.register('simple_classifier')
class JigsawPredictor(Predictor):	
	def predict(self, sentence: str) -> JsonDict:
		# This method is implemented in the base class.
		return self.predict_json({"sentence": sentence})

	def _json_to_instance(self, json_dict: JsonDict) -> Instance:
		sentence = json_dict["sentence"]
		return self._dataset_reader.text_to_instance(sentence)
		
	def dump_line(self, outputs: JsonDict) -> str:
		index_prediction = np.argmax(outputs['probs'])
		pred = self._model.vocab.get_token_from_index(index_prediction, namespace='target_labels')
		return f'{pred}\n'
