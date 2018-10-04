import collections

from torch import nn

from .transformer import Transformer
from .language_model_head import LanguageModelHead
from .similarity_head import SimilarityHead
from .multiple_choice_head import MultipleChoiceHead
from .classification_head import ClassificationHead


class DoubleHeadModel(nn.Module):
    """ Transformer with language model and task specific heads """
    def __init__(self, cfg, clf_token, task_head_type, vocab=40990, n_ctx=512):
        super(DoubleHeadModel, self).__init__()
        self.transformer = Transformer(cfg, vocab=vocab, n_ctx=n_ctx)
        self.lm_head = LanguageModelHead(self.transformer, cfg)
        if isinstance(task_head_type, str):
            if task_head_type == 'MultipleChoice':
                self.task_head = MultipleChoiceHead(clf_token, cfg)
            elif task_head_type == 'DocumentSimilarity':
                self.task_head = SimilarityHead(clf_token, cfg)
            elif task_head_type == 'DocumentClassification':
                # the three classes correspond to entailment, contradiction and neutral.
                self.task_head = ClassificationHead(clf_token, cfg, 3)
            else:
                raise ValueError("task_head_type is expected to be 'MultipleChoice' "
                                 "'DocumentSimilarity', 'DocumentClassification' or ('classification', n_class) "
                                 f"got {task_head_type}.")
        elif isinstance(task_head_type, collections.abc.Sequence) and len(task_head_type) == 2 and \
             task_head_type[0] == 'classification':
            n_class = task_head_type[1]
            self.task_head = ClassificationHead(clf_token, cfg, n_class)
        else:
            raise ValueError("task_head_type is expected to be 'MultipleChoice' "
                             "'DocumentSimilarity', 'DocumentClassification' or ('classification', n_class) "
                             f"got {task_head_type}.")

    def forward(self, x):
        h = self.transformer(x)
        lm_logits = self.lm_head(h)
        task_logits = self.task_head(h, x)

        return lm_logits, task_logits