import os
import csv
import logging
import json
from abc import ABC, abstractmethod
from PIL import ImageFile
from typing import Any, List, Tuple, Optional, Dict, Union

import torch
import skimage.io as io
from skimage.color import rgba2rgb, gray2rgb
from transformers import PreTrainedTokenizerBase
from torch.utils.data import Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Twitter201XInfo:
    """Container of all basic INFO from a Twitter-201{5, 7} example.

    Attributes:
        id: id (index) of example.
        label: sentiment label in {"0", "1", "2"}.
        image_fn: basename of image/meme.
        targetless_tweet: tweet text with target of interest
            substituted w/ "$T$".
        target: the target as it appears on the tweet.
    """

    def __init__(
        self,
        id: str,
        label: str,
        image_bn: str,
        targetless_tweet: str,
        target: str,
    ):
        """Init.

        See attributes.
        """
        # [1:-1] to remove self and var
        for var in self.__init__.__code__.co_varnames[1:-1]:
            setattr(self, var, locals()[var])

    def __str__(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())

    def __repr__(self) -> str:
        return self.__str__()


class Twitter201XDataset(Dataset, ABC):
    """Pytorch base dataset of Twitter-201{5, 7} dataset.

    Attributes:
        kind: "train", "dev" or "test".
        dir: tweets directory.
        name: something to reflect which dataset is being used.
        image_dir: memes directory.
        tokenizer: tokenizer of Transformer model this is going to be
            used for.
        label_mapping: mapping from labels read from dataset files to
            integer labels.
        data_class: class of output data.
        data: collection of data ready to be used in ML models.
        ids: IDs indexing `data`.
        entity_linker: entity linker of targets if args provided.
        entity_descriptions: Wikipedia descriptions of target entities
            if linker set up.
        entities_filename: where entities are (to be) stored.
        entity_data: stores return values of entity linker.
        cache: function to save entity linker's response for caching.
        logger: logging module.
        argparse_args: arguments for argparse.
    """

    fail_image_bn = "17_06_4705.jpg"

    argparse_args = dict(
        dir=dict(required=True, type=str, help="tweet dataset directory"),
        image_dir=dict(
            type=str, help="tweet dataset image directory (if not conventional)"
        ),
        train_split=dict(
            required=True,
            type=str,
            nargs="+",
            help="train dataset split(s)",
        ),
        dev_split=dict(
            type=str,
            nargs="+",
            help="development dataset split(s)",
        ),
        test_split=dict(
            type=str,
            nargs="+",
            help="test dataset split(s)",
        ),
    )

    def __init__(
        self,
        dir: str,
        kind: Union[str, List[str]],
        tokenizer: PreTrainedTokenizerBase,
        image_dir: Optional[str] = None,
        logging_level: Optional[int] = None,
        **encode_kwargs,
    ):
        """Init.

        Args:
            dir: tweets directory.
            kind: from {"train", "dev", "test"}.
            tokenizer: tokenizer of bert-based model.
            image_dir: memes directory, default to `dir`_images
            logging_level: level of severity of logger.
            entity_linker_kwargs: {"root_dir", "wiki_version", "threshold"}
                for entity linker. Default is no entity linking.
            encoder_kwargs: arguments specifically for `encode_plus`.
        """

        self.logger = logging.getLogger(__name__)
        if not logging_level:
            logging_level = logging.WARNING
        self.logger.setLevel(logging_level)

        self.to_tensor = transforms.ToTensor()

        self.kind = kind
        if isinstance(self.kind, str):
            self.kind = [self.kind]

        self.dir = dir
        self.name = os.path.basename(dir) + "(" + ",".join(self.kind) + ")"
        if image_dir is None:
            image_dir = os.path.normpath(dir) + "_images"
        self.image_dir = image_dir

        self.tokenizer = tokenizer

        file_lines = self._read_tsv()
        examples = self._parse_lines(file_lines)

        labels = set([example.label for example in examples])
        # sort for reproducibility + consistency between splits
        self.label_mapping = {l: i for i, l in enumerate(sorted(labels))}

        self.data_class = self.define_data_struct()

        self.data = self.encode_plus(
            examples,
            **encode_kwargs,
        )
        self.ids = list(self.data)

    @property
    def text_tokenizer(self):
        """Defines the text tokenizer (primarily used to allow inheriting
        of methods to datasets that use e.g. ViltProcessor, where to access
        the text tokenizer you need to use the `.tokenizer` property)."""
        return self.tokenizer

    def __len__(self):
        return len(self.data)

    @abstractmethod
    def define_data_struct(self):
        """Defines the structure that holds the final data."""

    @abstractmethod
    def encode_plus(self, examples, **encode_kwargs):
        """Prepares the inputs of the neural nets."""

    # removed init_entity_linker, _link_entity, and entity_integration

    def load_image(self, example: Twitter201XInfo) -> Tuple[torch.Tensor, bool]:
        """Loads and transforms image.

        Args:
            example: dataset example to load image for

        Returns:
            A tuple `(image, err)` where `image` is a transformed
            image and err indicates whether an error occurred while
            loading the actual image (and therefore a default image
            is used)
        """
        try:
            image_filename = os.path.join(self.image_dir, example.image_bn)
            image = io.imread(image_filename)

            if image.shape[-1] == 4:
                image = rgba2rgb(image)
            elif len(image.shape) < 3 or image.shape[-1] == 1:
                image = gray2rgb(image)

            image = self.to_tensor(image).to(torch.float32)
            err = False
        except:
            image_filename = os.path.join(self.image_dir, self.fail_image_bn)
            image = io.imread(image_filename)

            if image.shape[-1] == 4:
                image = rgba2rgb(image)
            elif len(image.shape) < 3 or image.shape[-1] == 1:
                image = gray2rgb(image)

            image = self.to_tensor(image).to(torch.float32)
            err = True

        return image, err

    def _read_tsv(self) -> List[Any]:
        """Read tsv file designated by attrs `dir` and `kind`.

        Returns:
            A list of read lines.
        """
        lines = []
        for kind in self.kind:
            with open(os.path.join(self.dir, kind + ".tsv")) as fp:
                reader = csv.reader(fp, delimiter="\t")
                # get rid of headers, should be:
                #   ['index', '#1 Label', '#2 ImageID', '#3 String', '#3 String']
                next(reader)

                lines.extend([line for line in reader])

        return lines

    def _parse_lines(self, lines: List[Any]) -> List[Twitter201XInfo]:
        """Parses lines from tsv into `Twitter201XInfo`s.

        Args:
            lines: lines read from tsv.

        Returns:
            A list of examples.
        """
        examples = [Twitter201XInfo(*line) for line in lines]
        return examples
