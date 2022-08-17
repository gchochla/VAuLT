from typing import Optional, Union

from transformers.training_args import TrainingArguments
from torch.utils.data import Dataset

from vault.tmsc_utils.trainer import Twitter201XTrainer
from .model import TomBertWithResNetForTMSC


class TomBertTrainerForTMSC(Twitter201XTrainer):
    """Trainer class fom `TomBertWithResNet` model.

    Attributes:
        train_image_encoder: whether the image encoder
            is being trained.
        See `Twitter201XTrainer`.
    """

    def __init__(
        self,
        model: TomBertWithResNetForTMSC,
        dataset: Dataset,
        train_args: TrainingArguments,
        dev_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        logging_level: Optional[Union[int, str]] = None,
    ):
        super().__init__(
            model, dataset, train_args, dev_dataset, test_dataset, logging_level
        )

        self.train_image_encoder = self.model.train_image_encoder

    def train_init(self):
        super().train_init()
        if not self.train_image_encoder:
            self.model.resnet.eval()

    def input_batch_kwargs(self, batch):
        (
            _,
            input_ids,
            input_mask,
            type_ids,
            target_input_ids,
            target_input_mask,
            target_type_ids,
            image,
            _,
        ) = batch

        return dict(
            input_ids=input_ids,
            target_input_ids=target_input_ids,
            images=image,
            token_type_ids=type_ids,
            target_type_ids=target_type_ids,
            attention_mask=input_mask,
            target_attention_mask=target_input_mask,
            return_embeddings=not self.train_image_encoder,
        )

    def get_logits_from_model(self, return_vals, batch, data_loader, epoch=0):
        ids, *_ = batch

        if not self.train_image_encoder:
            logits, embeddings = return_vals
            if epoch == 0:
                data_loader.dataset.replace_images_with_embeddings(
                    {
                        _id.item(): embedding
                        for _id, embedding in zip(ids, embeddings.to("cpu"))
                    }
                )
            return logits

        return return_vals

    def eval_end(self, data_loader=None):
        if not self.train_image_encoder:
            self.model.resnet.eval()

    # def init_optimizer(self) -> torch.optim.Optimizer:
    #     return torch.optim.AdamW(
    #         self.model.parameters(),
    #         lr=self.exp_handler.learning_rate,
    #         betas=[self.exp_handler.adam_beta1, self.exp_handler.adam_beta2],
    #         eps=self.exp_handler.adam_epsilon,
    #         weight_decay=self.exp_handler.weight_decay,
    #     )
