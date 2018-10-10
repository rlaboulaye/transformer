import collections

from torch import nn

from .transformer import Transformer
from .language_model_head import LanguageModelHead
from .similarity_head import SimilarityHead
from .multiple_choice_head import MultipleChoiceHead
from .classification_head import ClassificationHead


class DoubleHeadModel(nn.Module):
    """ Transformer with language model and task specific heads """
    def __init__(self, cfg, clf_token, task, vocab=40990, n_ctx=512):
        super(DoubleHeadModel, self).__init__()
        task_type = task['task_type']
        self.transformer = Transformer(cfg, vocab=vocab, n_ctx=n_ctx)
        self.lm_head = LanguageModelHead(self.transformer, cfg)
        if task_type == 'MultipleChoice':
            self.task_head = MultipleChoiceHead(clf_token, cfg)
        elif task_type == 'DocumentSimilarity':
            self.task_head = SimilarityHead(clf_token, cfg)
        elif task_type == 'DocumentClassification':
            # the three classes correspond to entailment, contradiction and neutral.
            assert('num_classes' in task['target'])
            self.task_head = ClassificationHead(clf_token, cfg, task['target']['num_classes'])
        else:
            raise ValueError("task_type is expected to be 'MultipleChoice' "
                             "'DocumentSimilarity', 'DocumentClassification' "
                             f"got {task_type}.")

    def forward(self, x):
        h = self.transformer(x)
        lm_logits = self.lm_head(h)
        task_logits = self.task_head(h, x)

        return lm_logits, task_logits