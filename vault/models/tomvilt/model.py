import logging
from typing import Optional, Union, Tuple
from copy import deepcopy

import torch
import torch.nn as nn
from transformers import PretrainedConfig, AutoConfig, BertModel

from vault.models.vault import VaultForTMSC, VaultModel
from vault.modules import BertCrossEncoder, ResNetEmbeddings
from vault.utils import extend_invert_attention_mask


class TomViltForTMSC(nn.Module):
    """Target-oriented Multimodal Vilt from
    `Adapting BERT for target-oriented sentiment classification`
    (https://www.ijcai.org/proceedings/2019/0751.pdf)

    Attributes:
        n_classes: the number of output classes.
        use_tweet_bert: whether to use tweet Bert before Vilt.
        tweet_bert: bert model for the tweet sequence if `use_tweet_bert`,
            else `None`.
        target_bert: bert model for solely the target.
        visual2textual_embeddings_mapper: mapping from the space of
            visual features to the space of textual features.
        target2image_attention: Bert-based encoder that queries visual
            features based on a target representation.
        multimodal_attention: [Bert]Vilt.
        logger: logging module.
    """

    argparse_args = dict(
        model_name_or_path=dict(
            default="bert-base-uncased",
            type=str,
            help="model to load into Bert parts of model",
        ),
        vilt_model_name_or_path=dict(
            default="dandelin/vilt-b32-mlm",
            type=str,
            help="model to use for Vilt parts of model",
        ),
        mm_pooling=dict(
            default="first",
            choices=["first", "cls", "both"],
            type=str,
            help="what pooling to use for multimodal encoder",
        ),
        vilt_dropout_prob=dict(
            default=0.1,
            type=float,
            help="Vault/Vilt internal dropout probability",
        ),
        use_tweet_bert=dict(
            action="store_true", help="essentially whether to use Vault or Vilt"
        ),
        vis_emb_dim=dict(
            default=2048, type=int, help="dimension of input visual embeddings"
        ),
    )

    def __init__(
        self,
        bert_config: PretrainedConfig,
        vilt_config: PretrainedConfig,
        n_classes: int = 3,
        vis_emb_dim: int = 2048,
        vilt_dropout_prob: float = 0.1,
        use_tweet_bert: bool = True,
        logging_level: Optional[Union[int, str]] = None,
    ):
        """Init.

        Args:
            bert_config: Bert configuration.
            vilt_config: Vilt configuration.
            n_classes: the number of output classes (in the paper that's 3).
            vis_emb_dim: dimension of visual representations per region.
            dropout_prob: dropout probability for Vilt (bcs original is 0).
            use_tweet_bert: whether to use `tweet_bert` before Vilt.
            logging_level: level of severity of logger.
        """

        super().__init__()

        self.n_classes = n_classes
        self.use_tweet_bert = use_tweet_bert

        # these should return BaseModelOutputWithPoolingAndCrossAttentions
        # in author implementation they return (output, pooled_output),
        # we can grab these by .last_hidden_layer, .pooler_output
        self.target_bert = BertModel(bert_config)

        self.visual2textual_embeddings_mapper = nn.Linear(
            vis_emb_dim, bert_config.hidden_size
        )

        self.target2image_attention = BertCrossEncoder(bert_config)

        self.multimodal_attention = VaultForTMSC(
            vilt_config,
            bert_config=bert_config if self.use_tweet_bert else None,
            n_classes=n_classes,
            logging_level=logging_level,
            vilt_dropout_prob=vilt_dropout_prob,
        )

        self.logger = logging.getLogger(__name__)
        if not logging_level:
            logging_level = logging.WARNING
        self.logger.setLevel(logging_level)

    @classmethod
    def from_pretrained(
        cls,
        bert_pretrained_name_or_path,
        vilt_pretrained_name_or_path,
        *model_args,
        **kwargs,
    ):
        """Hacky way to initialize stuff with Bert and Vilt weights."""
        bert_config = AutoConfig.from_pretrained(bert_pretrained_name_or_path)

        kwargs["dropout_prob"] = kwargs.get(
            "dropout_prob", bert_config.hidden_dropout_prob
        )

        try:
            vilt_config = AutoConfig.from_pretrained(
                vilt_pretrained_name_or_path
            )
            model = cls(bert_config, vilt_config, *model_args, **kwargs)
        except:
            vilt_config = AutoConfig.from_pretrained("dandelin/vilt-b32-mlm")
            model = cls(bert_config, vilt_config, *model_args, **kwargs)

        bert = BertModel.from_pretrained(bert_pretrained_name_or_path)
        bert_state_dict = bert.state_dict()
        vilt = VaultModel.from_pretrained(
            vilt_pretrained_name_or_path,
            bert_pretrained_name_or_path if model.use_tweet_bert else None,
        )

        model.multimodal_attention.load_state_dict(
            vilt.state_dict(), strict=False
        )  # false for classifier

        if (
            kwargs.get("tie_target_bert_weights", False)
            and model.use_tweet_bert
        ):
            model.target_bert = model.multimodal_attention.bert
        else:
            model.target_bert.load_state_dict(bert_state_dict)

        bert_encoder_state_dict = {
            ".".join(k.split(".")[1:]): v
            for k, v in bert_state_dict.items()
            if k.startswith("encoder")
        }

        model.target2image_attention.load_state_dict(bert_encoder_state_dict)

        model.logger.debug(
            "Initialized, target_bert, multimodal_attention (with"
            f"{'out' if not model.use_tweet_bert else ''} tweet bert)"
            " and target2image_attention from pretrained Bert & Vilt weights"
        )

        return model

    def resize_token_embeddings(self, tokenizer_length):
        self.multimodal_attention.resize_token_embeddings(tokenizer_length)
        self.target_bert.resize_token_embeddings(tokenizer_length)

    def get_input_embeddings(self):
        return {
            "tweet": self.multimodal_attention.get_input_embeddings(),
            "target": self.target_bert.get_input_embeddings(),
        }

    def set_input_embeddings(self, value):
        self.multimodal_attention.set_input_embeddings(value["tweet"])
        self.target_bert.set_input_embeddings(value["target"])

    def forward(
        self,
        input_ids: torch.Tensor,
        target_input_ids: torch.Tensor,
        visual_embeddings: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        target_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        target_attention_mask: Optional[torch.Tensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        tie_target_bert_weights: Optional[bool] = False,
    ) -> torch.Tensor:
        """Forward propagation.

        Passes target through Bert. Passes tweet+target through (potentially
        another) Bert. Maps visual embeddings to the same dimension as text
        embeddings. Filters/transforms visual embeddings by querying them
        with target representation. Concatenates pooled visual representation
        (no sequence tokens where defined for this, e.g. anologous to CLS)
        with pooled tweet+target representation. Passes through (multimodal)
        Bert. Produces final logits.

        Args:
            input_ids: bert-based model input IDs for targetless tweet
                followed by the target.
            target_input_ids: bert-based model input IDs for the target.
            visual_embeddings: image embeddings per region
                (batch x region x embedding).
            token_type_ids: type IDs for `input_ids`.
            target_type_ids: type IDs for `target_input_ids`.
            attention_mask: attention mask for `input_ids`.
            target_attention_mask: attention mask for `target_input_ids`.
            tie_target_bert_weights: copy tweet+target weights to target
                Bert.

        Returns:
            Logits.
        """

        n_regions = visual_embeddings.size(1)
        self.logger.debug(f"Image regions: {n_regions}")

        if tie_target_bert_weights and self.use_tweet_bert:
            self.target_bert = deepcopy(self.tweet_bert)

        ### Target output
        target_output = self.target_bert(
            input_ids=target_input_ids,
            attention_mask=target_attention_mask,
            token_type_ids=target_type_ids,
        ).last_hidden_state

        self.logger.debug(f"Target output shape: {target_output.shape}")

        ### Grab image part of mm mask
        if image_attention_mask is not None:
            ei_image_mask = extend_invert_attention_mask(
                image_attention_mask, next(self.parameters()).dtype
            )
            self.logger.debug(
                f"Extended and inverted mm attention mask: {ei_image_mask.shape}"
            )
        else:
            ei_image_mask = None

        ### Map vision to text features
        visual_embeddings = self.visual2textual_embeddings_mapper(
            visual_embeddings
        )  # batch x n_regions x bert hidden dim

        self.logger.debug(
            f"Visual embeddings shape after mapping: {visual_embeddings.shape}"
        )

        ### Apply target attention to visual features
        visual_embeddings = self.target2image_attention(
            querying_hidden_states=target_output,
            queried_hidden_states=visual_embeddings,
            attention_mask=ei_image_mask,
            output_all_encoded_layers=False,
        )

        self.logger.debug(
            f"Visual embeddings shape after attention: {visual_embeddings.shape}"
        )

        self.logger.debug(f"Multimodal text mask shape: {attention_mask.shape}")
        self.logger.debug(
            f"Multimodal text token IDs shape: {token_type_ids.shape}"
        )
        self.logger.debug(
            f"Multimodal image mask shape: {target_attention_mask.shape}"
        )

        logits = self.multimodal_attention(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            image_embeds=visual_embeddings,
            pixel_mask=target_attention_mask,
        )

        return logits


class TomViltWithResNetForTMSC(TomViltForTMSC):
    """Simple 'wrapper' around `TomViltForTMSC` with ResNet as visual encoder.

    Attributes:
        See `TomVilt`.
        resnet: pretrained ResNet that outputs embeddings per region.
        train_image_encoder: whether to fine-tune ResNet.
    """

    argparse_args = deepcopy(TomViltForTMSC.argparse_args)
    argparse_args.pop("vis_emb_dim")
    argparse_args.update(
        dict(
            resnet_depth=dict(
                default=101,
                type=int,
                help="which resnet to use",
            ),
            train_image_encoder=dict(
                action="store_true", help="whether to train ResNet"
            ),
        )
    )

    def __init__(
        self,
        bert_config: PretrainedConfig,
        vilt_config: PretrainedConfig,
        n_classes: int = 3,
        resnet_depth: int = 152,
        dropout_prob: float = 0.1,
        use_tweet_bert: bool = True,
        train_image_encoder: bool = False,
        logging_level: Optional[Union[int, str]] = None,
    ):
        """Init.

        Args:
            See `TomVilt` except for `vis_emb_dim`.
            resnet_depth: which resnet to use, defaults to 152
                (basically depth).
            train_image_encoder: whether to finetune ResNet.
                If `False`, sets `requires_grad` to `False`
                and uses `torch.no_grad()`.
        """
        super().__init__(
            bert_config,
            vilt_config,
            n_classes,
            ResNetEmbeddings.resnet_out_dim[resnet_depth],
            dropout_prob,
            use_tweet_bert,
            logging_level=logging_level,
        )
        self.resnet = ResNetEmbeddings(resnet_depth)

        self.train_image_encoder = train_image_encoder
        if not train_image_encoder:
            self.resnet.freeze_image_encoder()

    # TODO: check the default of tie_target_bert_weights in original implementation
    def forward(
        self,
        input_ids: torch.Tensor,
        target_input_ids: torch.Tensor,
        images: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        target_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        target_attention_mask: Optional[torch.Tensor] = None,
        tie_target_bert_weights: torch.Tensor = False,
        return_embeddings: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward propagation.

        Args:
            See `TomVilt` except for `visual_embeddings`
                and `image_attention_mask`.
            images: Images (of same size, hence the Tensor). This yields
                the same amount of regions in each. If embeddings are available,
                they can alternatively be directly provided (no training of
                the image encoder therefore actually occurs).
            return_embeddings: whether to also return embeddings.

        Returns:
            Logits or tuple of logits and visual embeddings if `return_embeddings`
            is True.
        """

        visual_embeddings = self.resnet(images, train=self.train_image_encoder)

        if attention_mask is not None:
            n_regions = visual_embeddings.size(1)
            vision_mask = torch.ones(
                input_ids.size(0), n_regions, device=attention_mask.device
            )
        else:
            vision_mask = None

        logits = super().forward(
            input_ids,
            target_input_ids,
            visual_embeddings,
            token_type_ids,
            target_type_ids,
            attention_mask,
            target_attention_mask,
            vision_mask,
            tie_target_bert_weights,
        )

        if return_embeddings:
            return (logits, visual_embeddings)
        return logits
