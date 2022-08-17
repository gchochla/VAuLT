import logging
from copy import deepcopy
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
from transformers import BertModel, PretrainedConfig, AutoModel
from transformers.models.bert.modeling_bert import BertEncoder

from vault.modules import BertCrossEncoder, BertPoolerDim, ResNetEmbeddings
from vault.utils import extend_invert_attention_mask


class TomBertForTMSC(nn.Module):
    """Target-oriented Multimodal Bert from
    `Adapting BERT for target-oriented sentiment classification`
    (https://www.ijcai.org/proceedings/2019/0751.pdf)

    Attributes:
        n_classes: the number of output classes (in the paper that's 3).
        pooling: pooling method of the final multimodal representation.
        tweet_bert: bert model for the tweet sequence.
        target_bert: bert model for solely the target.
        visual2textual_embeddings_mapper: mapping from the space of
            visual features to the space of textual features.
        target2image_attention: Bert-based encoder that queries visual
            features based on a target representation.
        target2image_pooler: pooler of `target2image_attention`.
        multimodal_attention: Bert encoder (expected to receive concat'd
            visual and textual representations).
        pool_and_predict: pooler of `multimodal_attention` plus mapping to
            output (logit) space.
        logger: logging module.
    """

    argparse_args = dict(
        model_name_or_path=dict(
            default="bert-base-uncased",
            type=str,
            help="model to load into Bert parts of model",
        ),
        tweet_model_name_or_path=dict(
            type=str,
            help="model to use for tweet bert",
        ),
        num_hidden_cross_layers=dict(
            default=1,
            type=int,
            help="number of layers in cross-attention "
            "(targeted attention on visual features)",
        ),
        pooling=dict(
            default="first",
            choices=["first", "cls", "both"],
            type=str,
            help="what pooling to use for multimodal encoder",
        ),
        vis_emb_dim=dict(
            default=2048, type=int, help="dimension of input visual embeddings"
        ),
    )

    def __init__(
        self,
        config: str,
        n_classes: int = 3,
        pooling: str = "first",
        vis_emb_dim: int = 2048,
        logging_level: Optional[Union[int, str]] = None,
        **kwargs,
    ):
        """Init.

        Args:
            config: transformer configuration.
            n_classes: the number of output classes (in the paper that's 3).
            pooling: pooling method of the final multimodal representation.
            vis_emb_dim: dimension of visual representations per region.
            logging_level: level of severity of logger.
        """

        super().__init__()

        self.n_classes = n_classes
        self.pooling = pooling.lower()

        # these should return BaseModelOutputWithPoolingAndCrossAttentions
        # in author implementation they return (output, pooled_output),
        # we can grab these by .last_hidden_layer, .pooler_output
        self.tweet_bert = BertModel(config)
        self.target_bert = BertModel(config)

        self.visual2textual_embeddings_mapper = nn.Linear(
            vis_emb_dim, config.hidden_size
        )

        self.target2image_attention = BertCrossEncoder(config)
        self.target2image_pooler = BertPoolerDim(config)

        self.multimodal_attention = BertEncoder(config)

        dropout_prob = getattr(
            config, "out_dropout_prob", config.hidden_dropout_prob
        )

        if pooling == "cls":
            self.pool_and_predict = nn.Sequential(
                BertPoolerDim(config, 1),
                nn.Dropout(dropout_prob),
                nn.Linear(config.hidden_size, n_classes),
            )
        elif pooling == "first":
            self.pool_and_predict = nn.Sequential(
                BertPoolerDim(config),
                nn.Dropout(dropout_prob),
                nn.Linear(config.hidden_size, n_classes),
            )
        else:  # both
            self.pool_and_predict = nn.Sequential(
                BertPoolerDim(config, [0, 1]),
                nn.Flatten(),
                nn.Dropout(dropout_prob),
                nn.Linear(config.hidden_size * 2, n_classes),
            )

        self.logger = logging.getLogger(__name__)
        if not logging_level:
            logging_level = logging.WARNING
        self.logger.setLevel(logging_level)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        pretrained_tweet_model_name_or_path=None,
        *model_args,
        **kwargs,
    ):
        """Hacky way to initialize stuff with Bert (and only Bert) weights."""
        config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path)

        num_hidden_cross_layers = kwargs.get("num_hidden_cross_layers", None)
        set_cross_layers = (
            num_hidden_cross_layers is not None
            and num_hidden_cross_layers != config.num_hidden_layers
        )
        if set_cross_layers:
            config.num_hidden_cross_layers = kwargs["num_hidden_cross_layers"]
            kwargs.pop("num_hidden_cross_layers")

        model = cls(config, *model_args, **kwargs)
        bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        bert_state_dict = bert.state_dict()

        if pretrained_tweet_model_name_or_path is not None:
            tweet_bert = AutoModel.from_pretrained(
                pretrained_tweet_model_name_or_path
            )
            model.tweet_bert = tweet_bert
            model.target_bert = deepcopy(tweet_bert)
        else:
            model.tweet_bert.load_state_dict(bert_state_dict)
            model.target_bert.load_state_dict(bert_state_dict)

        if kwargs.get("tie_target_bert_weights", False):
            model.target_bert = model.tweet_bert

        bert_encoder_state_dict = {
            ".".join(k.split(".")[1:]): v
            for k, v in bert_state_dict.items()
            if k.startswith("encoder")
        }
        model.multimodal_attention.load_state_dict(bert_encoder_state_dict)
        model.target2image_attention.load_state_dict(
            bert_encoder_state_dict, strict=not set_cross_layers
        )

        model.logger.debug(
            "Initialized tweet_bert, target_bert, multimodal_attention"
            " and target2image_attention from pretrained Bert weights"
        )

        return model

    def resize_token_embeddings(self, tokenizer_length):
        self.tweet_bert.resize_token_embeddings(tokenizer_length)
        self.target_bert.resize_token_embeddings(tokenizer_length)

    def get_input_embeddings(self):
        return {
            "tweet": self.tweet_bert.get_input_embeddings(),
            "target": self.target_bert.get_input_embeddings(),
        }

    def set_input_embeddings(self, value):
        self.tweet_bert.set_input_embeddings(value["tweet"])
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
        mm_attention_mask: Optional[torch.Tensor] = None,
        tie_target_bert_weights: bool = False,
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
            mm_attention_mask: multimodal attention mask.
            tie_target_bert_weights: copy tweet+target weights to target
                Bert.

        Returns:
            Logits.
        """

        n_regions = visual_embeddings.size(1)
        self.logger.debug(f"Image regions: {n_regions}")

        if tie_target_bert_weights:
            self.target_bert = deepcopy(self.tweet_bert)

        ### Tweet + Target output
        tweet_output = self.tweet_bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state

        self.logger.debug(f"Tweet output shape: {tweet_output.shape}")

        ### Target output
        target_output = self.target_bert(
            input_ids=target_input_ids,
            attention_mask=target_attention_mask,
            token_type_ids=target_type_ids,
        ).last_hidden_state

        self.logger.debug(f"Target output shape: {target_output.shape}")

        ### Grab image part of mm mask
        if mm_attention_mask is not None:
            ei_image_mask = extend_invert_attention_mask(
                mm_attention_mask[:, :n_regions], next(self.parameters()).dtype
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

        # It's kinda weird to only keep 1 token before the multimodal bert
        visual_embeddings = self.target2image_pooler(visual_embeddings)

        self.logger.debug(
            f"Visual embeddings shape after pooling: {visual_embeddings.shape}"
        )

        ### Create multimodal embeddings
        mm_embeddings = torch.cat(
            (visual_embeddings.unsqueeze(1), tweet_output), dim=1
        )

        self.logger.debug(f"Multimodal input shape: {mm_embeddings.shape}")

        ### Grab multimodal mask
        if mm_attention_mask is not None:
            ei_mm_mask = extend_invert_attention_mask(
                # since we pooled one visual token, we keep only 1 of the 49 token masks
                mm_attention_mask[:, (n_regions - 1) :],
                next(self.parameters()).dtype,
            )
            self.logger.debug(
                f"Extended and inverted mm attention mask: {ei_mm_mask.shape}"
            )
        else:
            ei_mm_mask = None

        ### Run multimodal encoder
        mm_output = self.multimodal_attention(
            hidden_states=mm_embeddings, attention_mask=ei_mm_mask
        ).last_hidden_state

        self.logger.debug(f"Final representation shape: {mm_output.shape}")

        ### Get logits
        logits = self.pool_and_predict(mm_output)

        return logits


class TomBertWithResNetForTMSC(TomBertForTMSC):
    """Simple 'wrapper' around `TomBertForTMSC` with ResNet as visual encoder.

    Attributes:
        See `TomBert`.
        resnet: pretrained ResNet that outputs embeddings per region.
        train_image_encoder: whether to fine-tune ResNet.
    """

    argparse_args = deepcopy(TomBertForTMSC.argparse_args)
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
        config: PretrainedConfig,
        n_classes: int = 3,
        pooling: str = "first",
        resnet_depth: int = 152,
        train_image_encoder: bool = False,
        logging_level: Optional[Union[int, str]] = None,
    ):
        """Init.

        Args:
            See `TomBert` except for `vis_emb_dim`.
            resnet_depth: which resnet to use, defaults to 152
                (basically depth).
            train_image_encoder: whether to finetune ResNet.
                If `False`, sets `requires_grad` to `False`
                and uses `torch.no_grad()`.
        """

        super().__init__(
            config,
            n_classes,
            pooling,
            ResNetEmbeddings.resnet_out_dim[resnet_depth],
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
            See `TomBert` except for `visual_embeddings`
                and `mm_attention_mask`.
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
            mm_attention_mask = torch.cat((vision_mask, attention_mask), dim=1)
        else:
            mm_attention_mask = None

        logits = super().forward(
            input_ids,
            target_input_ids,
            visual_embeddings,
            token_type_ids,
            target_type_ids,
            attention_mask,
            target_attention_mask,
            mm_attention_mask,
            tie_target_bert_weights,
        )

        if return_embeddings:
            return (logits, visual_embeddings)
        return logits
