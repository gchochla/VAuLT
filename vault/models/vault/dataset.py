from typing import Optional, List, Dict, Union, Iterable, Tuple, Any, Callable
from copy import deepcopy

import torch
from torchvision import transforms
from recordclass import RecordClass, asdict
from transformers import ViltProcessor

from vault.tmsc_utils.dataset import Twitter201XDataset, Twitter201XInfo
from vault.vl_utils.dataset import BloombergTwitterCorpus, MVSA
from .utils import vilt_safe_image_preprocess, relative_random_crop


class VaultDatasetForBloombergTwitterCorpus(BloombergTwitterCorpus):
    """Bloomberg Twitter text-image dataset."""

    argparse_args = deepcopy(BloombergTwitterCorpus.argparse_args)
    argparse_args.pop("crop_size")
    argparse_args.update(
        dict(
            max_length=dict(
                default=40,
                type=int,
                help="Max tokenized length of input text",
            )
        )
    )

    def __init__(
        self,
        root_dir: Union[List[str], str],
        splits: Union[List[str], str],
        processor: ViltProcessor,
        max_length: int,
        tasks: Union[str, List[str]] = "text_is_represented",
        twitter_preprocessor: Optional[Callable] = None,
        demojizer: Optional[Callable] = None,
        image_augmentation: bool = False,
        logging_level: Optional[int] = None,
        _dev_size: Optional[int] = None,
        _test_size: Optional[int] = None,
    ):
        """Init.

        Args:
            Check `BloombergTwitterCorpus` except for
                `encode_kwargs` and `crop_size`.
            max_length: max tokenized length of text.
        """
        super().__init__(
            root_dir=root_dir,
            splits=splits,
            processor=processor,
            encode_kwargs=dict(max_length=max_length),
            crop_size=None,
            tasks=tasks,
            twitter_preprocessor=twitter_preprocessor,
            demojizer=demojizer,
            image_augmentation=image_augmentation,
            logging_level=logging_level,
            _dev_size=_dev_size,
            _test_size=_test_size,
        )

    def _init_image_preprocessor(self, crop_size):
        return transforms.Compose(
            [vilt_safe_image_preprocess(), transforms.ToTensor()]
        )

    def _init_image_transformation(self, crop_size):
        return relative_random_crop()


class VaultDatasetForMVSA(MVSA):
    """MVSA dataset."""

    argparse_args = deepcopy(MVSA.argparse_args)
    argparse_args.pop("crop_size")
    argparse_args.update(
        dict(
            max_length=dict(
                default=40,
                type=int,
                help="Max tokenized length of input text",
            )
        )
    )

    def __init__(
        self,
        root_dir: Union[List[str], str],
        splits: Union[List[str], str],
        processor: ViltProcessor,
        max_length: int,
        twitter_preprocessor: Optional[Callable] = None,
        demojizer: Optional[Callable] = None,
        preprocessed: bool = True,
        image_augmentation: bool = False,
        logging_level: Optional[int] = None,
    ):
        """Init.

        Args:
            Check `MVSA` except for `encode_kwargs` and `crop_size`.
            max_length: max tokenized length of text.
        """

        super().__init__(
            root_dir=root_dir,
            splits=splits,
            processor=processor,
            encode_kwargs=dict(max_length=max_length),
            crop_size=None,
            twitter_preprocessor=twitter_preprocessor,
            demojizer=demojizer,
            logging_level=logging_level,
            preprocessed=preprocessed,
            image_augmentation=image_augmentation,
        )

    def _init_image_preprocessor(self, crop_size):
        return vilt_safe_image_preprocess()

    def _init_image_transformation(self, crop_size):
        return relative_random_crop()


class VaultDatasetForTMSC(Twitter201XDataset):
    """Pytorch dataset of Twitter-201{5, 7} dataset. Contains
    the necessary sequences for ViLT, the tweet without the
    target concat'd with the target, and the image.

    Attributes:
        See `Twitter201XDataset`.
        image_transformation: preprocessing and transformation of PIL
            image to Pytorch tensor.
        preprocess_on_fetch: whether to perform transformation of
            image on fetch and otherwise retain the original image
            (allows for augmentation).
        max_length: ViLT's maximum # text tokens.
    """

    max_length = 40

    argparse_args = deepcopy(Twitter201XDataset.argparse_args)
    argparse_args.update(
        dict(
            max_length=dict(
                default=max_length,
                type=int,
                help="Max total tokenized length of tweet and target",
            ),
            crop_size=dict(
                default=224, type=int, help="dimension to crop images to"
            ),
            preprocess_on_fetch=dict(
                action="store_true",
                help="whether to preprocess images when fetching them to user, "
                "aka augmentation",
            ),
        )
    )

    def __init__(
        self,
        dir: str,
        kind: str,
        max_length: int,
        tokenizer: ViltProcessor,
        crop_size: int,
        image_dir: Optional[str] = None,
        logging_level: Optional[int] = None,
        entity_linker_kwargs: Optional[Dict[str, Any]] = None,
        preprocess_on_fetch: Optional[bool] = True,
    ):
        """Init.

        Args:
            See `Twitter201XDataset`.
            max_length: max length of text input.
            preprocess_on_fetch: whether to perform transformation of
                image on fetch and otherwise retain the original image
                (allows for augmentation).
        """
        self.image_transformation = self._get_image_transform(crop_size)
        self.preprocess_on_fetch = preprocess_on_fetch

        assert max_length <= self.max_length  # ViLT constraint

        super().__init__(
            dir,
            kind,
            tokenizer,
            image_dir=image_dir,
            logging_level=logging_level,
            entity_linker_kwargs=entity_linker_kwargs,
            **{"max_length": max_length},
        )

    @property
    def text_tokenizer(self):
        return self.tokenizer.tokenizer

    def define_data_struct(self):
        """Defines `data` structure."""

        class Twitter201XData(RecordClass):
            """Container of all basic DATA from a Twitter-201{5, 7} example
            for ViLT.

            Attributes:
                id: id (index) of example.
                input_ids: bert-based model input IDs for targetless tweet
                    followed by the target.
                input_mask: corresponding attention mask.
                type_ids: corresponding type IDs.
                target_input_ids: bert-based model input IDs for the target.
                target_input_mask: corresponding attention mask.
                target_type_ids: corresponding type IDs.
                image: image/meme.
                label_id: integer label.
            """

            id: int
            input_ids: List[int]
            text_mask: List[int]
            type_ids: List[int]
            image: torch.Tensor
            image_mask: torch.Tensor
            label_id: int

        return Twitter201XData

    def encode_plus(
        self, examples: List[Twitter201XInfo], max_length: int
    ) -> Dict[int, "Twitter201XData"]:
        """Encodes input sequences to IDs.

        Args:
            examples: basic info from Twitter-201{5, 7} dataset.
            max_total_length: max length for targetless tweet and
                target sequence.
            max_target_length: max length for target sequence.

        Returns:
            A dict whose keys are the example IDs and the value the
            corresponding basic data from Twitter-201{5, 7} dataset.
        """
        err_cnt = 0

        # used only for logging
        max_dataset_total_length = 0

        data = {}

        for example in examples:

            text = (
                example.targetless_tweet
                + self.text_tokenizer.special_tokens_map["sep_token"]
                + example.target
            )

            ### update length value
            tokens = self.text_tokenizer.tokenize(text)
            max_dataset_total_length = max(
                max_dataset_total_length, len(tokens)
            )
            ###

            image, error = self.load_image(example)
            if error:
                err_cnt += 1

            if self.preprocess_on_fetch:
                input = self.text_tokenizer.encode_plus(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    return_tensors="pt",
                )

                data[int(example.id)] = self.data_class(
                    id=int(example.id),
                    input_ids=input["input_ids"][0],
                    text_mask=input["attention_mask"][0],
                    type_ids=input.get("token_type_ids", [None])[0],
                    image=image,
                    image_mask=None,
                    label_id=torch.tensor(self.label_mapping[example.label]),
                )
            else:
                input = self.tokenizer(
                    self.image_transformation(image),
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    return_tensors="pt",
                )

                data[int(example.id)] = self.data_class(
                    id=int(example.id),
                    input_ids=input["input_ids"][0],
                    text_mask=input["attention_mask"][0],
                    type_ids=input.get("token_type_ids", [None])[0],
                    image=input["pixel_values"][0],
                    image_mask=input["pixel_mask"][0],
                    label_id=torch.tensor(self.label_mapping[example.label]),
                )

        self.logger.info(f"*** Example Data ***\n{next(iter(data.values()))}\n")
        msg = f"{err_cnt} errors occured whilst loading images"
        if err_cnt > 0:
            self.logger.warning(msg)
        else:
            self.logger.info(msg)
        self.logger.info(f"Max total length: {max_dataset_total_length}")

        return data

    def __getitem__(self, index: int) -> Tuple[torch.Tensor]:
        """See `Twitter201XData` attribute order to get return order."""
        # NOTE: this returns ID too.

        datum = self.data[self.ids[index]]

        if not self.preprocess_on_fetch:
            return tuple(asdict(datum).values())

        image_input = self.tokenizer.feature_extractor(
            self.image_transformation(datum.image), return_tensors="pt"
        )
        image = image_input["pixel_values"][0]
        image_mask = image_input["pixel_mask"][0]

        return tuple(
            [
                val
                if k not in ("image", "image_mask")
                else image
                if k == "image"
                else image_mask
                for k, val in asdict(datum).items()
            ]
        )

    def collate_fn(
        self, batch: Tuple[Tuple[torch.Tensor, List[List[int]], None]]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[List[List[int]]], None]:

        ret = []
        for val in zip(*batch):
            try:
                batched_val = torch.stack(val)
            except:
                batched_val = val

            if any([v is None for v in batched_val]):
                batched_val = None

            ret.append(batched_val)

        return tuple(ret)

    def _get_image_transform(
        self, crop_size: Union[int, Iterable[int]]
    ) -> transforms.Compose:
        """Creates the necessary transformation of the image for CNNs
        to produce the same number of regions across images.

        Args:
            crop_size: size of sides to crop image to (up to 2 values).

        Returns:
            Transformation function from PIL to pytorch tensor.
        """

        # Things I see wrong with original image transform
        # (`https://github.com/jefferyYu/TomBERT/blob/31bc79fb9a913a2480d5a56ffe5009d986d6eb2a/run_multimodal_classifier.py#L293`):
        #   * random crop without resizing?
        #   * why random crop if augmentation is not used?
        #   * why random horizontal flip in images that might have text

        identity_transform = transforms.Lambda(lambda x: x)

        crop_transform = (
            transforms.Compose(
                [
                    transforms.Resize(crop_size + 32),
                    transforms.RandomCrop(crop_size),
                ]
            )
            if crop_size is not None
            else identity_transform
        )

        return crop_transform
