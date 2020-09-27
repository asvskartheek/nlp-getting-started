from src.models.simple_model import SimpleClassifier
from typing import Dict

import torch
from allennlp.data import Vocabulary
from allennlp.data import TextFieldTensors
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.modules.attention import Attention
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy

from allennlp.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from allennlp.nn.util import get_text_field_mask

@Model.register('attention_classifier')
class AttentionModel(Model):
	def __init__(self, vocab: Vocabulary, embedder: TextFieldEmbedder, encoder: Seq2VecEncoder, attention: Attention, **kwargs):
		super().__init__(vocab)

		self.embedder = embedder
		self.encoder = encoder
		num_labels = vocab.get_vocab_size("target_labels")
		self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
		self.accuracy = CategoricalAccuracy()

		self.attention = attention

	def __self_attention(self, embedded:torch.Tensor):
		attention_embeddings = torch.empty_like(embedded)
		for i,batch in enumerate(embedded):
			for word in batch:
				word = word.unsqueeze(0)
				wts = self.attention(word, embedded[i])
				attention_embeddings[i] = torch.mm(wts, embedded[i])
		return attention_embeddings

	def forward(self, text: TextFieldTensors, target: torch.Tensor=None, id:torch.Tensor=None, **kwargs):
		embedded = self.embedder(text)
		attention_embedded = self.__self_attention(embedded)

		if isinstance(self.encoder, CnnEncoder):
			mask = get_text_field_mask(text)
			encoded = self.encoder(attention_embedded, mask)
		else:
			encoded = self.encoder(attention_embedded)
		# encoded: (b, encoded_dim)
		logits = self.classifier(encoded)
		# logits: (b, num_labels)
		output = {'logits': logits}
		# Shape: (batch_size, num_labels)
		probs = torch.nn.functional.softmax(logits, dim=-1)
		output['probs'] = probs
		# Shape: (b,)
		if target is not None:
			loss = torch.nn.functional.cross_entropy(logits, target)
			self.accuracy(logits, target)
			output['loss'] = loss
			# Shape: (1,)
		return output

	def get_metrics(self, reset: bool = False, **kwargs) -> Dict[str, float]:
		return {"accuracy": self.accuracy.get_metric(reset)}
		