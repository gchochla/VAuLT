from typing import List, Tuple, Union

import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import (
    BertAttention,
    BertIntermediate,
    BertOutput,
)
from transformers import PretrainedConfig
from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)

from vault.utils import set_parameter_requires_grad


class BertCrossAttention(BertAttention):
    def forward(
        self,
        querying_hidden_states: torch.Tensor,
        queried_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        """Forward Propagation.

        If `encoder_hidden_states` is set, it is instead used
        for key and values, while `encoder_attention_mask`
        overwrites attention mask.

        Args:
            querying_hidden_states: the representation used as queries.
            queried_hidden_states: the representation used as keys and values.
            attention_mask: corresponding attention mask.

        Returns:
            Tuple where first element is output of Bert attention block and
            the rest possibly contain attention probabilities and past key
            values.
        """

        return super().forward(
            hidden_states=querying_hidden_states,
            encoder_hidden_states=queried_hidden_states,
            encoder_attention_mask=attention_mask,
        )


class BertCrossAttentionLayer(nn.Module):
    """Equivalent of BertLayer for target-guided attention
    of visual features.

    Attributes:
        attention: cross attention module.
        intermediate: intermediate module (linear to
            intermediate dimension + actf)
        output: output module (linear from intermediate
            space to input/output space, layer norm and
            dropout)
    """

    def __init__(self, config):
        """Init.

        Args:
            config: transformer configuration.
        """
        super().__init__()
        self.attention = BertCrossAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        querying_hidden_states: torch.Tensor,
        queried_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward propagation.

        Args:
            querying_hidden_states: the representation used as queries.
            queried_hidden_states: the representation used as keys and values.
            attention_mask: corresponding attention mask.

        Returns:
            Output of Bert block.
        """
        # [0] because transformers appends intermediate values
        attention_output = self.attention(
            querying_hidden_states, queried_hidden_states, attention_mask
        )[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# NOTE: could use BertEncoder directly but that has self-attention on query first
#   through BertLayer. [Not anymore: However, here we are changing the attetion mask as well.]
class BertCrossEncoder(nn.Module):
    """Equivalent of BertEncoder for target-guided attention
    of visual features.

    Attributes:
        layers: nn.ModuleList containing `BertCrossAttentionLayer`s.
    """

    def __init__(self, config: PretrainedConfig):
        """Init.

        Args:
            config: transformer configuration.
        """
        super().__init__()

        if not hasattr(config, "num_hidden_cross_layers"):
            config.num_hidden_cross_layers = config.num_hidden_layers

        self.layer = nn.ModuleList(
            [
                BertCrossAttentionLayer(config)
                for _ in range(config.num_hidden_cross_layers)
            ]
        )

    def forward(
        self,
        querying_hidden_states: torch.Tensor,
        queried_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_all_encoded_layers: bool = True,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Forward propagation. NOTE: only the querying hidden
        states get updated from layer to layer in the implementation.

        Args:
            querying_hidden_states: the representation used as queries.
            queried_hidden_states: the representation used as keys and values.
            attention_mask: corresponding attention mask.
            output_all_encoded_layers: whether to return intermediate values
                after each transformer block, defaults to `True`.

        Returns:
            List of transformer block outputs if `output_all_encoded_layers`
            or just the output of the entire module otherwise.
        """

        # For consistrency with BertEncoder, I extend and invert in TomBert
        # ei_attention_mask = extend_invert_attention_mask(
        #     attention_mask, next(self.parameters()).dtype
        # )

        all_encoder_layers = []
        for layer_module in self.layer:
            querying_hidden_states = layer_module(
                querying_hidden_states, queried_hidden_states, attention_mask
            )
            if output_all_encoded_layers:
                all_encoder_layers.append(querying_hidden_states)
        if not output_all_encoded_layers:
            return querying_hidden_states
        return all_encoder_layers


class BertPoolerDim(nn.Module):
    """Equivalent of transformers.models.bert.modeling_bert.BertPooler
    with extra `token` argument to allow for pooling from other dimensions.

    Attrs:
        token: which token to pool from.
        dense: linear layer with identical input and output dimensions + tanh.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        tokens: Union[int, List[int]] = 0,
    ):
        """Init.

        Args:
            config: transformer configuration.
            token: which token to pool from.
        """
        super().__init__()
        self.tokens = tokens
        self.dense = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size), nn.Tanh()
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Pools the model by simply taking the hidden state
        corresponding to the `token`-th token.

        Args:
            hidden_states: hidden states of some Bert layer.

        Returns:
            The pooled hidden state.
        """
        token_tensor = hidden_states[:, self.tokens]
        pooled_output = self.dense(token_tensor)
        return pooled_output


class ResNetEmbeddings(nn.Module):
    """ResNet feature extractor.
    Features are extracted before Average Pooling layer of ResNets.
    Note that requires_grad is kept as `True` (because we use
    feature extractors as submodules in classifiers) and
    torch.no_grad() should be used to disable differentiation.

    Attributes:
        features: net doing the feature extraction.
        resnet_ids_512: ResNet IDs that result in 512-dimensional
            representations.
        resnet_ids_2048: Same for 2048.
        resnet_out_dim: mapping from resnet IDs (int or str) to
            output dim.
    """

    resnet_ids_512 = [18, 34]
    resnet_ids_2048 = [50, 101, 152]
    resnet_out_dim = {id: 512 for id in resnet_ids_512}
    resnet_out_dim.update({str(id): 512 for id in resnet_ids_512})
    resnet_out_dim.update({id: 2048 for id in resnet_ids_2048})
    resnet_out_dim.update({str(id): 2048 for id in resnet_ids_2048})

    def __init__(self, resnet_id: Union[int, str] = 152):
        """Init.
        Args:
            resnet_id: ResNet version to load,
                default is 152 (i.e. ResNet152).
        """

        super().__init__()

        resnet_ids = self.resnet_ids_512 + self.resnet_ids_2048
        assert resnet_id in resnet_ids + list(map(str, resnet_ids))

        self.resnet_id = resnet_id

        resnet = globals()["resnet{}".format(resnet_id)](
            pretrained=True, progress=False
        )
        # thankfully resnet's forward is just a series of forward propagations
        # https://github.com/pytorch/vision/blob/39772ece7c87ab38b9c2b9df8e7c85e967a739de/torchvision/models/resnet.py#L264
        self.features = nn.Sequential(*list(resnet.children())[:-2])

    @property
    def id(self):
        return self.resnet_id

    @property
    def out_dim(self):
        return self.resnet_out_dim[self.resnet_id]

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation.

        Args:
            x: input images.

        Returns:
            512-dimensional region representations if
            ResNet{18, 34}, 2048-dimensional otherwise,
            shape is #batches x #regions x repr_dim.
        """

        x = self.features(x)
        # -2 bcs last two represent grid of regions
        # permute to get regions before embedding dim
        x = x.flatten(-2).permute(0, 2, 1)
        return x

    def forward(self, x: torch.Tensor, train: bool = True) -> torch.Tensor:
        """Forward propagation with check for training
        and whether image or already computeed embedding
        is provided.

        Args:
            x: input images or already computed embeddings.

        Returns:
            512-dimensional region representations if
            ResNet{18, 34}, 2048-dimensional otherwise,
            shape is #batches x #regions x repr_dim.
        """

        if x.ndim == 4:  # if actual images passed
            if train:
                x = self._forward(x)
            else:
                with torch.no_grad():
                    x = self._forward(x)
        return x

    def freeze_image_encoder(self):
        """Freeze ResNet."""
        set_parameter_requires_grad(self, False)

    def unfreeze_image_encoder(self):
        """Unfreeze ResNet."""
        set_parameter_requires_grad(self, True)
