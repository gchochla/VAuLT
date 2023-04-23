from abc import ABC
import logging
from typing import Optional, Union, Dict, List, Tuple, Any

import torch
import torch.nn as nn
from transformers import (
    PretrainedConfig,
    AutoModel,
    ViltModel,
    ViltForMaskedLM,
    ViltForQuestionAnswering,
    ViltForImagesAndTextClassification,
    ViltForImageAndTextRetrieval,
)

from vault.utils import set_parameter_requires_grad


class VaultMixin(nn.Module, ABC):
    """A mixin to implement VAuLT by inheriting FIRST from
    this and then the desired ViLT class.

    Attributes:
        bert: language model.
        freeze_lm: whether to freeze language model (`bert`).
    """

    argparse_args = dict(
        vilt_model_name_or_path=dict(
            default="dandelin/vilt-b32-mlm",
            type=str,
            help="model to load into Vilt parts of model",
        ),
        bert_model_name_or_path=dict(
            type=str,
            help="model to load into Bert parts of model, if any",
        ),
        vilt_dropout_prob=dict(
            default=0.1,
            type=float,
            help="dropout in internal Vilt layers",
        ),
        freeze_lm=dict(
            action="store_true", help="whether to freeze language model"
        ),
        use_vilt_position_embeddings=dict(
            action="store_true",
            help="whether to use Vilt's position embeddings",
        ),
    )

    def __init__(
        self,
        vilt_config,
        bert_config: Optional[PretrainedConfig] = None,
        freeze_lm: bool = False,
        vilt_dropout_prob: float = 0.0,
        use_vilt_position_embeddings: bool = False,
        *args,
        **kwargs,
    ):
        """Init.

        Args:
            ViLT-related args.
            bert_config: configuration of Bert. Bert is not used
                if not provided (default).
        """

        # these are 0 in the pretrained Vilt models, 0.1 in Berts
        vilt_config.t_prob = vilt_dropout_prob
        vilt_config.attention_probshidden_dropou_dropout_prob = (
            vilt_dropout_prob
        )

        # if language model is used, skip ViLT's position embeddings
        if bert_config is not None and not use_vilt_position_embeddings:
            setattr(vilt_config, "position_embedding_type", "NOT_absolute")

        super().__init__(vilt_config, *args, **kwargs)
        self.bert = (
            AutoModel.from_config(config=bert_config, add_pooling_layer=False)
            if bert_config is not None
            else None
        )

        self.freeze_lm = freeze_lm
        if self.bert is not None and freeze_lm:
            set_parameter_requires_grad(self.bert, False)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_vilt: str,
        pretrained_bert: Optional[str] = None,
        freeze_lm: bool = False,
        use_vilt_position_embeddings: bool = False,
        *args,
        **kwargs,
    ):
        """Extends `from_pretrained` to possibly initialize
        the Bert model as well.

        Args:
            pretrained_vilt: vilt model name or path.
            pretrained_bert: bert model name or path, if any.
            model_args, kwargs: vilt params.
        """
        model = super().from_pretrained(pretrained_vilt, *args, **kwargs)

        # if language model is used, can skip ViLT's position embeddings
        if pretrained_bert is not None and not use_vilt_position_embeddings:
            model.embeddings.text_embeddings.position_embedding_type = (
                "NOT_absolute"
            )

        model.bert = (
            AutoModel.from_pretrained(pretrained_bert, add_pooling_layer=False)
            if pretrained_bert is not None
            else None
        )

        model.freeze_lm = freeze_lm
        if model.bert is not None and freeze_lm:
            set_parameter_requires_grad(model.bert, False)

        return model

    def resize_token_embeddings(self, tokenizer_length):
        """Extends `resize_token_embeddings` to include Bert."""
        if self.bert is not None:
            self.bert.resize_token_embeddings(tokenizer_length)
        else:
            super().resize_token_embeddings(tokenizer_length)

    def get_input_embeddings(self):
        """Extends `get_input_embeddings` to include Bert."""
        if self.bert is not None:
            return self.bert.get_input_embeddings()
        else:
            return super().get_input_embeddings()

    def set_input_embeddings(self, value):
        """Extends `resize_token_embeddings` to include Bert."""
        if self.bert is not None:
            self.bert.set_input_embeddings(value)
        else:
            super().set_input_embeddings(value)

    def lm_preprocess(
        self, *args, **kwargs
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Preprocess text using LM if it has been selected
        (Overwrites input IDs and instead provides input embeds).

        Returns:
            Potentially overwritten args and kwargs.
        """
        if self.bert is not None:
            input_ids = kwargs.get(
                "input_ids", args[0] if len(args) > 0 else None
            )
            attention_mask = kwargs.get(
                "attention_mask", args[1] if len(args) > 1 else None
            )
            token_type_ids = kwargs.get(
                "token_type_ids", args[2] if len(args) > 2 else None
            )
            inputs_embeds = kwargs.get(
                "inputs_embeds", args[6] if len(args) > 6 else None
            )

            if (
                self.bert.embeddings.token_type_embeddings.num_embeddings < 2
                and token_type_ids is not None
            ):
                token_type_ids = torch.zeros_like(
                    token_type_ids, device=token_type_ids.device
                )

            bert_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
            )

            with torch.set_grad_enabled(not self.freeze_lm):
                inputs_embeds = self.bert(**bert_kwargs).last_hidden_state

            if len(args) > 0:
                args[0] = None
            else:
                kwargs["input_ids"] = None

            if len(args) > 6:
                args[6] = inputs_embeds
            else:
                kwargs["inputs_embeds"] = inputs_embeds

        return args, kwargs

    def vilt_forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Forward propagation. If Bert has been selected,
        then text is first processed by Bert and then the
        contextual embeddings are passed to Vilt, otherwise
        text is processed only by Vilt.

        Args, Returns:
            See corresponding ViLT class.
        """

        args, kwargs = self.lm_preprocess(*args, **kwargs)
        return self.vilt_forward(*args, **kwargs)


class PipelineVaultMixin(VaultMixin):
    def __init__(
        self,
        vilt_config,
        bert_config: Optional[PretrainedConfig] = None,
        vilt_dropout_prob: float = 0,
        *args,
        **kwargs,
    ):
        vilt_device_id = kwargs.pop("vilt_device_id", None)
        lm_device_id = kwargs.pop("lm_device_id", None)
        output_device_id = kwargs.pop("output_device_id", None)

        self.inner_batch_size = kwargs.pop("inner_batch_size", None)

        super().__init__(
            vilt_config, bert_config, vilt_dropout_prob, *args, **kwargs
        )

        self.pipeline(vilt_device_id, lm_device_id, output_device_id)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        vilt_device_id = kwargs.pop("vilt_device_id", None)
        lm_device_id = kwargs.pop("lm_device_id", None)
        output_device_id = kwargs.pop("output_device_id", None)

        inner_batch_size = kwargs.pop("inner_batch_size", None)

        model = super().from_pretrained(*args, **kwargs)

        model.inner_batch_size = inner_batch_size
        PipelineVaultMixin.pipeline(
            model, vilt_device_id, lm_device_id, output_device_id
        )

        return model

    def pipeline(self, vilt_device_id, lm_device_id, output_device_id):
        self.pipelined = lm_device_id is not None and vilt_device_id is not None
        if self.pipelined:
            self.lm_device = f"cuda:{lm_device_id}"
            self.vilt_device = f"cuda:{vilt_device_id}"
            self.output_device = (
                None
                if output_device_id is None
                else "cpu"
                if output_device_id == "cpu"
                else f"cuda:{output_device_id}"
            )

            self = self.to(self.vilt_device, force=True)
            self.bert = self.bert.to(self.lm_device)

        else:
            self.lm_device = self.vilt_device = None

    def _args_kwargs_to_device(self, args, kwargs, device):
        args = [arg.to(device) if arg is not None else None for arg in args]
        kwargs = {
            k: v.to(device) if v is not None else None
            for k, v in kwargs.items()
        }
        return args, kwargs

    def _args_kwargs_split(self, args, kwargs):
        try:
            batch_size = len(args[0])
        except IndexError:
            batch_size = len(next(iter(kwargs.values())))

        inner_batch_size = self.inner_batch_size or batch_size

        num_inner_batches = (
            batch_size + inner_batch_size - 1
        ) // inner_batch_size

        if args:
            args = list(
                zip(*[arg.split(inner_batch_size, dim=0) for arg in args])
            )
        else:
            args = [[] for _ in range(num_inner_batches)]

        if kwargs:
            kwargs_split_values = {
                k: v.split(inner_batch_size, dim=0) for k, v in kwargs.items()
            }
            kwargs = [
                {k: v[i] for k, v in kwargs_split_values.items()}
                for i in range(num_inner_batches)
            ]
        else:
            kwargs = [{} for _ in range(num_inner_batches)]

        return iter(zip(args, kwargs))

    def to(self, *args, **kwargs):
        force = kwargs.pop("force", False)
        if self.pipelined and not force:
            return self
        else:
            return super().to(*args, **kwargs)

    def lm_forward(self, *args, **kwargs):
        if self.pipelined:
            args, kwargs = self._args_kwargs_to_device(
                args, kwargs, self.lm_device
            )

        args, kwargs = super().lm_preprocess(*args, **kwargs)

        if self.pipelined:
            args, kwargs = self._args_kwargs_to_device(
                args, kwargs, self.vilt_device
            )

        return args, kwargs

    def forward(self, *args, **kwargs):
        if not self.pipelined:
            return super().forward(*args, **kwargs)

        batch_iter = self._args_kwargs_split(args, kwargs)
        ret = []

        inp_args, inp_kwargs = next(batch_iter)
        inter_args, inter_kwargs = self.lm_forward(*inp_args, **inp_kwargs)

        for inp_args, inp_kwargs in batch_iter:
            outs = self.vilt_forward(*inter_args, **inter_kwargs)
            ret.append(outs)

            inter_args, inter_kwargs = self.lm_forward(*inp_args, **inp_kwargs)

        outs = self.vilt_forward(*inter_args, **inter_kwargs)
        ret.append(outs)

        ret = {k: torch.concat([d[k] for d in ret], dim=0) for k in ret[0]}
        if self.output_device is not None:
            ret = self._args_kwargs_to_device([], ret, self.output_device)

        for k in ret:
            outs[k] = ret[k]

        return outs


class VaultModel(VaultMixin, ViltModel):
    """Vision and Augmented Language Transformer. Enables the use
    of a language model to preprocess the text. Check `VaultMixin`,
    `ViltModel` for details."""


class VaultForImageAndTextRetrieval(VaultMixin, ViltForImageAndTextRetrieval):
    """VAuLT for Image and Text retrieval, e.g. MSCOCO and F30K.
    Check `VaultMixin`, `ViltForImageAndTextRetrieval` for details."""

    def __init__(self, *args, **kwargs):
        """Init. If evoked from `from_pretrained` and conditions are met,
        it also adds the original itm_score head to the model to load
        the pre-trained weights."""

        from_pretrained = kwargs.pop("__from_pretrained__", False)
        super().__init__(*args, **kwargs)
        if from_pretrained:
            self.itm_score = nn.Sequential()
            self.itm_score.add_module(
                "fc", nn.Linear(self.config.hidden_size, 2)
            )

    @classmethod
    def from_pretrained(self, pretrained_vilt: str, *args, **kwargs):
        """Makes sure the original ITM head is used to initialize
        the rank output head if checkpoint doesn't do that."""
        from_pretrained = "itm" in pretrained_vilt  # what about finetuned here
        kwargs["__from_pretrained__"] = from_pretrained
        model = super().from_pretrained(pretrained_vilt, *args, **kwargs)

        if from_pretrained:
            model.rank_output.weight.data = model.itm_score.fc.weight.data[1:]
            model.rank_output.bias.data = model.itm_score.fc.bias.data[1:]
            del model.itm_score

        return model


class VaultForImagesAndTextClassification(
    VaultMixin, ViltForImagesAndTextClassification
):
    """VAuLT for (multiple) Images and Text Classification, e.g. NLVR2.
    Check `VaultMixin`, `ViltForImagesAndTextClassification` for details.
    Allows loading from base ViLTs, not just NLVR2-finetuned ones."""

    def __init__(self, config, *args, **kwargs):
        from_pretrained = kwargs.pop("__from_pretrained__", False)

        default_num_images = 2  # nlvr2
        num_images = kwargs.pop("num_images", None)

        if num_images is not None:
            config.num_images = num_images
        elif config.num_images == -1:
            config.num_images = default_num_images

        super().__init__(config, *args, **kwargs)

        if not from_pretrained:
            self.resize_token_type_embeddings()

    def resize_token_type_embeddings(self):
        """Resizes the token type embeddings to match the needs of the
        forward propagation based on how many images are used for the task.
        The pretrained token type embedding otherwise used for a single
        image are propagated to all the images."""

        if self.config.modality_type_vocab_size != self.config.num_images + 1:
            self.config.modality_type_vocab_size = self.config.num_images + 1
            pretrained_token_type_embeddings = (
                self.vilt.embeddings.token_type_embeddings.weight.data
            )
            assert len(pretrained_token_type_embeddings) == 2
            self.vilt.embeddings.token_type_embeddings = nn.Embedding(
                self.config.modality_type_vocab_size,
                self.vilt.config.hidden_size,
            )
            self.vilt.embeddings.token_type_embeddings.weight.data[
                0
            ] = pretrained_token_type_embeddings[0]
            self.vilt.embeddings.token_type_embeddings.weight.data[
                1:
            ] = pretrained_token_type_embeddings[1]

            # NOTE: image token types are 1, 2, ...
            # base pretrained vilt only 1
            # copy 1 to 2, ... also?

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        kwargs["__from_pretrained__"] = True
        model = super().from_pretrained(*args, **kwargs)
        VaultForImagesAndTextClassification.resize_token_type_embeddings(model)

        return model


class VaultForMaskedLM(VaultMixin, ViltForMaskedLM):
    """VAuLT for MLM. Check `VaultMixin`, `ViltForMaskedLM` for details."""


# num labels
class VaultForQuestionAnswering(VaultMixin, ViltForQuestionAnswering):
    """VAuLT for VQA, e.g. VQAv2. Check `VaultMixin`,
    `ViltForQuestionAnswering` for details.
    Allows loading from base ViLTs, not just VQAv2-finetuned ones."""

    def __init__(self, config, *args, **kwargs):
        num_labels = kwargs.pop("n_classes", None)
        super().__init__(config, *args, **kwargs)

        if num_labels is not None and num_labels != self.config.num_labels:
            self.renew_classifier(num_labels)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        num_labels = kwargs.pop("n_classes", None)
        model = super().from_pretrained(*args, **kwargs)

        if num_labels is not None and num_labels != model.config.num_labels:
            print(
                "Substituting current classifier, you should probably TRAIN "
                "this model on a down-stream task to be able to use it for "
                "predictions and inference."
            )
            VaultForQuestionAnswering.renew_classifier(model, num_labels)

        return model

    def renew_classifier(self, num_labels):
        curr_linear = self.classifier[-1]
        new_linear = torch.nn.Linear(
            curr_linear.in_features,
            num_labels,
            curr_linear.bias is not None,
        )
        new_linear.weight.data.normal_(mean=0, std=0.02)
        if new_linear.bias is not None:
            new_linear.bias.data.zero_()
        self.classifier[-1] = new_linear


class VaultForTMSC(VaultModel):
    """VAuLT for Target-oriented Multimodal Sentiment Classification.
    Includes an output classifier on top of its pooler.

    Attributes:
        See `BertViltModel`.
        classifier: Linear layer on top of Vilt pooler for final predictions.
        logger: logging module.
    """

    def __init__(
        self,
        vilt_config: PretrainedConfig,
        n_classes: int = 3,
        vilt_dropout_prob: float = 0.1,
        logging_level: Optional[Union[int, str]] = None,
        bert_config: Optional[PretrainedConfig] = None,
    ):
        """Init.

        Args:
            vilt_config: `ViltModel` config.
            n_classes: number of output classes.
            dropout_prob: Vilt & output dropout probability.
            logging_level: level to log at.
            bert_config: `BertModel` config to use, if any.
        """

        super().__init__(
            vilt_config,
            add_pooling_layer=True,
            bert_config=bert_config,
            vilt_dropout_prob=vilt_dropout_prob,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(vilt_dropout_prob),
            nn.Linear(self.config.hidden_size, n_classes),
        )

        self.logger = logging.getLogger(__name__)
        if not logging_level:
            logging_level = logging.WARNING
        self.logger.setLevel(logging_level)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward propagation. Pooled output is
        passed into final classifier.

        Args:
            See `ViltModel` args.

        Returns:
            Logits.
        """
        x = super().forward(*args, **kwargs)
        self.logger.debug(f"Output shape: {x.last_hidden_state.shape}")
        x = self.classifier(x.pooler_output).squeeze(-1)
        return x
