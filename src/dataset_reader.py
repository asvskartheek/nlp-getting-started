from typing import Dict, Iterable, List

import pandas as pd

from allennlp.data import DataLoader
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField, TextField, MetadataField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataloader import PyTorchDataLoader

# You can implement your own dataset reader by subclassing DatasetReader.
# At the very least, you need to implement the _read() method, preferably
# text_to_instance() as well.
@DatasetReader.register('classification-csv')
class ClassificationCsvReader(DatasetReader):
	def __init__(self,
				 tokenizer: Tokenizer = None,
				 token_indexers: Dict[str, TokenIndexer] = None,
				 max_tokens: int = None,
				 **kwargs):
		super().__init__(**kwargs)
		self.tokenizer = tokenizer or WhitespaceTokenizer()
		self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer(namespace='text')}
		self.max_tokens = max_tokens

	def text_to_instance(self, text: str,target: int=None, id:int=None, **kwargs) -> Instance:
		# Normalise text to lower-case
		tokens = self.tokenizer.tokenize(self._normalise(text))
		if self.max_tokens:
			tokens = tokens[:self.max_tokens]
		text_field = TextField(tokens, self.token_indexers)
		fields = {'text': text_field}
		if target:
			fields['target'] = LabelField(target, label_namespace='target_labels')
		if id:
			fields['id'] = MetadataField(id)
		return Instance(fields)

	def _read(self, file_path: str, **kwargs) -> Iterable[Instance]:
		df = pd.read_csv(file_path)
		for i, row in df.iterrows():
			text = row['text']

			if 'target' in df.columns:
				target = str(row['target'])
				id = row['id']
			else:
				target = None
				id = None
			
			# Ignore Keyword and Location
			yield self.text_to_instance(text, target, id)

	# to clean data
	def _normalise(self, text: str):
		text = text.lower() # lowercase
		text = text.replace(r"\#","") # replaces hashtags
		text = text.replace(r"http\S+","URL")  # remove URL addresses
		text = text.replace(r"@","")
		text = text.replace(r"[^A-Za-z0-9()!?\'\`\"]", " ")
		text = text.replace("\s{2,}", " ")
		return text
		