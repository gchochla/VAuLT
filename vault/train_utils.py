import logging
import tempfile
import operator
import functools
from typing import Optional, Any, Dict, Iterable
from dataclasses import dataclass, field

import torch
from transformers.training_args import TrainingArguments
from transformers.utils import add_start_docstrings


class EarlyStopping:
    """Implements early stopping in a Pytorch fashion, i.e. an init call
    where the model (you want to save) is an argument and a step function
    to be called after each evaluation.

    Attributes:
        model: nn.Module to be saved.
        tmp_fn: TemporaryFile, where to save model (can be None).
        patience: early stopping patience.
        cnt: number of early stopping steps that metric has not improved.
        delta: difference before new metric is considered better that the
            previous best one.
        higher_better: whether a higher metric is better.
        best: best metric value so far.
        best_<metric name>: other corresponding measurements can be passed
            as extra kwargs, they are stored when the main metric is stored
            by prepending 'best_' to the name.
        logger: logging module.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        patience: Optional[int],
        save_model: bool = False,
        delta: float = 0,
        higher_better: bool = False,
        logging_level: Optional[int] = None,
    ):
        """Init.

        Args:
            model: nn.Module to be saved.
            save_model: whether to save model.
            patience: early stopping patience, if `None then no early stopping.
            delta: difference before new metric is considered better that
                the previous best one.
            higher_better: whether a higher metric is better.
        """
        self.model = model
        self.tmp_fn = (
            tempfile.NamedTemporaryFile(mode="r+", suffix=".pt")
            if save_model
            else None
        )
        self.saved = False
        self.patience = patience
        self.cnt = 0
        self.delta = delta
        self.higher_better = higher_better

        self.best = None

        self.logger = logging.getLogger(__name__)
        if not logging_level:
            logging_level = logging.WARNING
        self.logger.setLevel(logging_level)

    def new_best(self, metric: float) -> bool:
        """Compares the `metric` appropriately to the current best.

        Args:
            metric: metric to compare best to.

        Returns:
            True if metric is indeed better, False otherwise.
        """
        if self.best is None:
            return True
        return (
            metric > self.best + self.delta
            if self.higher_better
            else metric < self.best - self.delta
        )

    def best_str(self) -> str:
        """Formats `best` appropriately."""
        if self.best is None:
            return "None"
        return f"{self.best:.6f}"

    def step(self, metric: Optional[float], **kwargs) -> bool:
        """Compares new metric (if it is provided) with previous best,
        saves model if so (and if `model_path` was not `None`) and
        updates count of unsuccessful steps.

        Args:
            metric: metric value based on which early stopping is used.
            kwargs: all desired metrics (including the metric passed).

        Returns:
            Whether the number of unsuccesful steps has exceeded the
            patience if patience has been set, else the signal to
            continue training (aka `False`).
        """
        if self.patience is None or metric is None:
            self._save()
            return False  # no early stopping, so user gets signal to continue

        if self.new_best(metric):
            self.logger.info(
                f"Metric improved: {self.best_str()} -> {metric:.6f}"
            )
            self._store_best(metric, **kwargs)
            self.cnt = 0
            self._save()
        else:
            self.cnt += 1
            self.logger.info(
                f"Patience counter increased to {self.cnt}/{self.patience}"
            )

        return self.cnt >= self.patience

    def _save(self):
        """Saves model and logs location."""
        if self.tmp_fn is not None:
            self.saved = True
            torch.save(self.model.state_dict(), self.tmp_fn.name)
            self.tmp_fn.seek(0)
            self.logger.info("Saved model to " + self.tmp_fn.name)

    def best_model(self) -> torch.nn.Module:
        """Loads last checkpoint (if any) and returns model."""
        if self.tmp_fn is not None and self.saved:
            state = torch.load(self.tmp_fn.name)
            self.model.load_state_dict(state)
        return self.model

    def _store_best(self, metric: float, **kwargs):
        """Saves best metric and potentially other corresponsing
        measurements in kwargs."""
        self.best = metric
        for key in kwargs:
            self.__setattr__("best_" + key, kwargs[key])

    def get_metrics(
        self,
    ) -> Optional[Dict[str, Any]]:
        """Returns accumulated best metrics.

        Returns:
            If the class was idle, nothing. Otherwise, if metrics were
            passed with kwargs in `step`, then these with the string
            `best_` prepended in their keys, else a generic dict
            with 'metric' as key and the best metric.
        """

        if self.best is None:
            return

        metrics = {
            k: v for k, v in self.__dict__.items() if k.startswith("best_")
        }

        if not metrics:
            metrics = {"metric": self.best}

        return metrics


def prod(args: Iterable[float]) -> float:
    return functools.reduce(operator.mul, args, 1)


@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class MyTrainingArguments(TrainingArguments):
    """
    Args:
        early_stopping_patience: early stopping patience, defaults to `None`
            (aka no early stopping).
        model_save: whether to save model using `torch.save`,
            defaults to `False`.
        spanemo_lca_coef: LCA loss coefficient, \in [0, 1]. `None` (default)
            denotes that LCA is not used.
        spanemo_lca_weighting: whether to use weights based on correlations
            for LCA terms.
        spanemo_lca_weighting_func: Function of normalized correlation
            to use for LCA terms
        multilabel_conditional_order: Order of relationship to model with
            `MultilabelConditionalWeights`, should be [0, 1].
            `None` (Default) denotes no such loss.
        multilabel_conditional_func: Function of probability to use for
            `MultilabelConditionalWeights`.
        correct_bias: if using AdamW from `transformers`, whether to
            correct bias, default is `False`.
        discard_classifier: if loading a local checkpoint, whether (not) to
            load final classifier.
    """

    early_stopping_patience: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Early stopping patience, `None` "
                "(default) denotes no early stopping"
            )
        },
    )

    model_save: bool = field(
        default=False,
        metadata={
            "help": (
                "whether to save model using `torch.save`,"
                " defaults to `False`."
            )
        },
    )

    model_load_filename: Optional[str] = field(
        default=None,
        metadata={
            "help": "filename to load local checkpoint from, default to `None`."
        },
    )

    spanemo_lca_coef: Optional[float] = field(
        default=None,
        metadata={
            "help": "SpanEmo's correlation-aware loss coefficient, "
            "should be [0, 1]. `None` (default) denotes no such loss."
        },
    )

    spanemo_lca_weighting: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to weigh the LCA terms "
            "according to correlation of pair"
        },
    )

    spanemo_lca_weighting_func: Optional[str] = field(
        default="dec_sqrt",
        metadata={
            "help": "Function of normalized correlation " "to use for LCA terms"
        },
    )

    multilabel_conditional_order: Optional[float] = field(
        default=None,
        metadata={
            "help": "Order of relationship to model with "
            "`MultilabelConditionalWeights`, should be [0, 1]. "
            "`None` (Default) denotes no such loss."
        },
    )

    multilabel_conditional_func: Optional[str] = field(
        default="dec_sqrt_p1",
        metadata={
            "help": "Function of probability to use for "
            "`MultilabelConditionalWeights`"
        },
    )

    correct_bias: bool = field(
        default=False,
        metadata={
            "help": "if using AdamW from `transformers`, whether to "
            "correct bias, default is `False`"
        },
    )

    discard_classifier: bool = field(
        default=False,
        metadata={
            "help": "if loading a local checkpoint, "
            "whether (not) to load final classifier"
        },
    )
