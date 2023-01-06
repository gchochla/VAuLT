from typing import Optional, Dict, List, Tuple, Union, Any
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

from vault.tmsc_utils.trainer import Twitter201XTrainer
from vault.vl_utils.trainer import VisionAndLanguageTrainer


class VaultTrainerForTMSC(Twitter201XTrainer):
    """Vault trainer for Target-oriented Multimodal Sentiment Classification.
    For attributes, check `Twitter201XTrainer`."""

    def input_batch_kwargs(self, batch):
        (
            _,
            input_ids,
            text_mask,
            type_ids,
            image,
            image_mask,
            _,
        ) = batch

        return dict(
            input_ids=input_ids,
            attention_mask=text_mask,
            token_type_ids=type_ids,
            pixel_values=image,
            pixel_mask=image_mask,
        )


class VaultTrainerForBloombergTwitterCorpus(VisionAndLanguageTrainer):
    early_stopping_metric = "eval_loss"

    def calculate_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, train: bool
    ) -> torch.Tensor:
        """Calculates loss based on predicted logits and labels.

        Args:
            logits: model predictions.
            labels: ground truth labels.
            train: whether this is during training.

        Returns:
            Loss.
        """
        criterion = nn.BCEWithLogitsLoss()
        return criterion(logits, labels)

    def get_logits_from_model(self, return_vals: Any, *args, **kwargs):
        """Grabs logits from output dict/struct."""
        return return_vals

    def get_eval_preds_from_batch(self, logits: torch.Tensor) -> List[int]:
        return (logits.sigmoid() >= 0.5).int().tolist()

    def get_eval_true_from_batch(self, labels: torch.Tensor) -> List[int]:
        return labels.int().tolist()

    def evaluation_metrics(
        self,
        eval_true: List[int],
        eval_preds: List[int],
        data_loader: DataLoader,
    ) -> Dict[str, float]:
        """Computes evaluation metrics (beyond evaluation loss).

        Args:
            eval_true: ground-truth labels.
            eval_preds: predictions.
            data_loader: DataLoader where data came from.

        Returns:
            A dict of metrics.
        """

        results = super().evaluation_metrics(eval_true, eval_preds, data_loader)
        _, _, f1_score, _ = precision_recall_fscore_support(
            eval_true, eval_preds, average="weighted", zero_division=0
        )
        results.update(dict(f1_score=f1_score))
        return results


class VaultTrainerForMVSA(VaultTrainerForBloombergTwitterCorpus):
    def __init__(self, *args, **kwargs):
        """See `VaultTrainerForBloombergTwitterCorpus`."""
        super().__init__(*args, **kwargs)
        self.preprocessed = self.dataset.preprocessed

    def get_eval_preds_from_batch(self, logits: torch.Tensor) -> List[int]:
        if self.preprocessed:
            return logits.argmax(-1).tolist()

        n_logits = logits.shape[-1]
        logit_groups = [
            logits[..., : n_logits // 2],
            logits[..., n_logits // 2 :],
        ]

        return [
            [tl.argmax(-1).tolist(), vl.argmax(-1).tolist()]
            for tl, vl in zip(*logit_groups)
        ]

    def calculate_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, train: bool
    ) -> torch.Tensor:
        """Calculates loss based on predicted logits and labels.
        Args:
            logits: model predictions.
            labels: ground truth labels.
            train: whether this is during training.
        Returns:
            Loss.
        """
        criterion = nn.CrossEntropyLoss()
        if self.preprocessed:
            return criterion(logits, labels)

        n_logits = logits.shape[-1]
        logit_groups = [
            logits[..., : n_logits // 2],
            logits[..., n_logits // 2 :],
        ]
        return 0.5 * (
            criterion(logit_groups[0], labels[..., 0])
            + criterion(logit_groups[1], labels[..., 1])
        )

    def evaluation_metrics(
        self,
        eval_true: List[List[int]],
        eval_preds: List[List[int]],
        data_loader: DataLoader,
    ) -> Dict[str, float]:
        """Computes evaluation metrics beyond eval loss. If MVSA is not preprocessed,
        separate metrics for text and images are computed (with "image_" and "text_"
        prefix in the names).

        Args:
            eval_true: ground-truth labels.
            eval_preds: predictions.
            data_loader: DataLoader where data came from.

        Returns:
            A dict of metrics.
        """

        def accuracy_f1(true, preds):
            acc = np.mean([pred == label for pred, label in zip(preds, true)])
            _, _, wf1, _ = precision_recall_fscore_support(
                true, preds, average="weighted", zero_division=0
            )
            _, _, micf1, _ = precision_recall_fscore_support(
                true, preds, average="micro", zero_division=0
            )
            _, _, macf1, _ = precision_recall_fscore_support(
                true, preds, average="macro", zero_division=0
            )
            return acc, macf1, micf1, wf1

        if not self.preprocessed:
            text_true, text_preds = [t[0] for t in eval_true], [
                t[0] for t in eval_preds
            ]
            img_true, img_preds = [t[1] for t in eval_true], [
                t[1] for t in eval_preds
            ]
            text_accuracy, text_macf1, text_micf1, text_wf1 = accuracy_f1(
                text_true, text_preds
            )
            img_accuracy, img_macf1, img_micf1, img_wf1 = accuracy_f1(
                img_true, img_preds
            )
            return dict(
                text_eval_accuracy=text_accuracy,
                image_eval_accuracy=img_accuracy,
                text_macro_f1_score=text_macf1,
                text_micro_f1_score=text_micf1,
                text_weighted_f1_score=text_wf1,
                image_macro_f1_score=img_macf1,
                image_micro_f1_score=img_micf1,
                image_weighted_f1_score=img_wf1,
            )
        else:
            eval_accuracy, macf1, micf1, wf1 = accuracy_f1(
                eval_true, eval_preds
            )
            return dict(
                eval_accuracy=eval_accuracy,
                macro_f1_score=macf1,
                micro_f1_score=micf1,
                weighted_f1_score=wf1,
            )


class VaultTrainerForImagesAndTextClassification(VisionAndLanguageTrainer):
    """Vault trainer for Images and Text classification. For attributes,
    check `VisionAndLanguageTrainer`."""


class VaultTrainerForQuestionAnswering(VisionAndLanguageTrainer):
    """Vault trainer for Question Answering. For attributes,
    check `VisionAndLanguageTrainer`."""

    def _get_labels_none(
        self, labels: Union[torch.Tensor, List[Optional[torch.Tensor]]]
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """Grabs labels that are not None and their indices."""
        inds = [i for i, l in enumerate(labels) if l is not None]
        labels = [labels[i] for i in inds]
        return labels, inds

    def calculate_loss(
        self,
        logits: torch.Tensor,
        labels: Union[torch.Tensor, List[Optional[torch.Tensor]]],
        train: bool,
    ) -> Optional[torch.Tensor]:
        """Calculates loss based on logits and labels.
        If all labels are None, nothing is returned.

        Args:
            logits: logit predictions.
            labels: ground-truth labels, in a tensor if all present
                or list of tensors and Nones if some are missing.

        Returns:
            Loss if any gt labels where present.
        """

        criterion = nn.BCEWithLogitsLoss()
        if not torch.is_tensor(labels):
            labels, inds = self._get_labels_none(labels)
            if not labels:
                return
            labels = torch.stack(labels)
            logits = logits[inds]

        # https://github.com/jnhwkim/ban-vqa/blob/54f044ce9020842b4cb69679e535f885bef57ca3/train.py#L19
        return criterion(logits, labels) * labels.size(1)

    def get_eval_true_from_batch(
        self, labels: Union[torch.Tensor, List[Optional[torch.Tensor]]]
    ) -> List[List[float]]:
        """Returns ground-truth labels from batch in list of lists, if any."""
        if torch.is_tensor(labels):
            return super().get_eval_true_from_batch(labels)
        else:
            labels, _ = self._get_labels_none(labels)
            return [l.tolist() for l in labels]

    def evaluation_metrics(
        self, eval_true: List[int], eval_preds: List[int], data_loader=None
    ) -> Dict[str, float]:
        """Computes evaluation metrics (beyond evaluation loss).

        In VQA, we have the scores of the answers, so we grab the score of the
        most liekly answer according to the model.

        Args:
            eval_true: ground-truth labels.
            eval_preds: predictions.

        Returns:
            A dict of metrics.
        """

        eval_accuracy = np.mean(
            # get score of each answer
            [label[pred] for pred, label in zip(eval_preds, eval_true)]
        )

        return dict(eval_accuracy=eval_accuracy)


class VaultTrainerForImageAndTextRetrieval(VisionAndLanguageTrainer):
    """Vault trainer for Images and Text classification. For attributes,
    check `VisionAndLanguageTrainer`."""

    def calculate_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        train: bool,
    ) -> Optional[torch.Tensor]:
        """Calculates loss based on logits and labels.

        Args:
            logits: logit predictions.
            labels: ground-truth labels.

        Returns:
            Loss.
        """

        criterion = nn.BCEWithLogitsLoss()
        return criterion(logits, labels)

    def evaluate(
        self,
        data_loader: DataLoader,
        tqdm_message: Optional[str] = "Evaluation",
    ):

        """Evaluates model on `data_loader`.

        Args:
            data_loader: dataset to evaluate on.
            tqdm_message: what to print if tqdm is used.
        """

        self.model.eval()
        self.eval_init(data_loader)

        dataset = data_loader.dataset

        batch_itr = (
            tqdm(
                dataset.all_image_text_pairs(),
                desc=tqdm_message,
                dynamic_ncols=True,
                total=len(dataset) * len(dataset.image_fns),
            )
            if not self.exp_handler.disable_tqdm
            else data_loader
        )

        eval_preds = []
        eval_true = []
        eval_loss = 0.0

        image_scores = defaultdict(dict)
        text_scores = defaultdict(dict)

        for batch in batch_itr:
            inputs, label, ids = batch
            inputs, label = self.batch_to_device([inputs, label])

            with torch.no_grad():
                return_vals = self.model(**inputs)

            logits = self.get_logits_from_model(return_vals)

            loss = self.calculate_loss(logits, label, train=False)
            eval_loss += loss.item()

            eval_preds.append(self.get_eval_preds_from_batch(logits))
            eval_true.append(self.get_eval_true_from_batch(label))

            score = logits.item()
            image_scores[ids["image_identifier"]][score] = label.cpu().item()
            text_scores[ids["text_identifier"]][score] = label.cpu().item()

        results = self.evaluation_metrics(
            eval_true, eval_preds, image_scores, text_scores
        )

        self.model.train()
        self.eval_end(data_loader)

        return results

    def get_eval_preds_from_batch(self, logits: torch.Tensor) -> List[int]:
        """Returns predictions in batch based on logits."""
        return (logits.sigmoid() >= 0.5).int().tolist()

    def evaluation_metrics(
        self,
        eval_true: List[List[int]],
        eval_preds: List[List[int]],
        image_scores: Dict[int, Dict[float, int]],
        text_scores: Dict[Tuple[int, int], Dict[float, int]],
    ) -> Dict[str, float]:
        """Computes evaluation metrics (beyond evaluation loss),
        including retrieval scores.

        Args:
            eval_true: ground-truth labels.
            eval_preds: predictions.

        Returns:
            A dict of metrics.
        """

        results = super().evaluation_metrics(eval_true, eval_preds)

        for kind, score_dict in zip(
            ["image", "text"], [image_scores, text_scores]
        ):
            # TODO: add retrieval k's to CosmicArgs
            retrieval_hits = {1: [], 5: [], 10: []}
            for scores in score_dict.values():
                retrieved_labels = [
                    scores[score] for score in sorted(scores, reverse=True)
                ]
                for k in retrieval_hits:
                    retrieval_hits[k].append(
                        any([label == 1 for label in retrieved_labels[:k]])
                    )

            results.update(
                {f"{kind}-R@{k}": np.mean(v) for k, v in retrieval_hits.items()}
            )

        return results
