import torch


def compute_language_model_loss(X, M, lm_logits, lm_criterion):
    X_shifted = X[:, :, 1:, 0].contiguous().view(-1)
    M = M.view(-1, M.size(-1))
    lm_losses = lm_criterion(lm_logits, X_shifted)
    lm_losses = lm_losses.view(X.size(0) * X.size(1), X.size(2) - 1)
    lm_losses = lm_losses * M[:, 1:]
    lm_losses = lm_losses.sum(1) / torch.sum(M[:, 1:], 1)
    return lm_losses.sum()

def compute_task_loss(Y, task_logits, task_criterion):
    task_losses = task_criterion(task_logits, Y)
    return task_losses.sum()

def compute_double_head_loss(X, Y, M, lm_logits, task_logits, lm_criterion, task_criterion, lm_coef, task_coef):
    lm_loss = 0
    if lm_coef != 0:
        lm_loss = compute_language_model_loss(X, M, lm_logits, lm_criterion)
    task_loss = 0
    if task_coef != 0:
        task_loss = compute_task_loss(Y, task_logits, task_criterion)
    double_head_loss = lm_coef * lm_loss + task_coef * task_loss
    return double_head_loss, lm_loss, task_loss

def compute_accuracy(Y, task_logits):
    predictions = task_logits.view(Y.shape[0], -1).argmax(-1)
    return (predictions == Y).double().mean()

class Evaluator:
    # TODO add logic in init to choose a preprocesser and scorer
    def __init__(self, lm_criterion, task_criterion, lm_coef, task_coef, target_type):
        self.lm_criterion = lm_criterion
        self.task_criterion = task_criterion
        self.lm_coef = lm_coef
        self.task_coef = task_coef
        self.target_type = target_type
        self.preprocess = lambda x: x
        self.scoring_metric = None

        if target_type == 'regression':
            self.preprocess = self._flatten
            self.scoring_metric = self._score_error
        elif target_type == 'classification':
            self.scoring_metric = self._score_acurracy


    def _flatten(self, input):
        return input.view(-1)

    def compute_language_model_loss(self, X, M, lm_logits):
        X_shifted = X[:, :, 1:, 0].contiguous().view(-1)
        M = M.view(-1, M.size(-1))
        lm_losses = self.lm_criterion(lm_logits, X_shifted)
        lm_losses = lm_losses.view(X.size(0) * X.size(1), X.size(2) - 1)
        lm_losses = lm_losses * M[:, 1:]
        lm_losses = lm_losses.sum(1) / torch.sum(M[:, 1:], 1)
        return lm_losses.sum()

    def compute_task_loss(self, Y, task_logits):
        # print(Y)
        input = self.preprocess(task_logits)
        task_losses = self.task_criterion(input, Y)
        return task_losses.sum()

    def compute_double_head_loss(self, X, Y, M, lm_logits, task_logits):
        lm_loss = 0
        if self.lm_coef != 0:
            lm_loss = self.compute_language_model_loss(X, M, lm_logits)
        task_loss = 0
        if self.task_coef != 0:
            task_loss = self.compute_task_loss(Y, task_logits)
        double_head_loss = self.lm_coef * lm_loss + self.task_coef * task_loss
        return double_head_loss, lm_loss, task_loss

    def _score_acurracy(self, Y, task_logits):
        predictions = task_logits.view(Y.shape[0], -1).argmax(-1)
        return (predictions == Y).double().mean()

    def _score_error(self, Y, task_output):
        return torch.nn.MSELoss(reduction="elementwise_mean")(task_output, Y)

    def compute_score(self, Y, task_logits):
        input = self.preprocess(task_logits)
        return self.scoring_metric(Y, input)
