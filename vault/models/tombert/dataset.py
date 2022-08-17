from typing import Iterable, List, Union, Optional, Tuple, Dict, Any
from copy import deepcopy

import torch
from torchvision import transforms
from recordclass import RecordClass, asdict
from transformers import PreTrainedTokenizerBase

from vault.tmsc_utils.dataset import Twitter201XDataset, Twitter201XInfo


class TomBertDatasetForTMSC(Twitter201XDataset):
    """Pytorch dataset of Twitter-201{5, 7} dataset. Contains
    the necessary sequences for TomBERT, aka the target tokenized
    on its own plus the tweet without the target concat'd with the
    target.

    Attributes:
        See `Twitter201XDataset`.
        image_transformation: preprocessing and transformation of PIL
            image to Pytorch tensor.
    """

    argparse_args = deepcopy(Twitter201XDataset.argparse_args)
    argparse_args.update(
        dict(
            max_total_length=dict(
                default=64,
                type=int,
                help="Max total tokenized length of tweet and target",
            ),
            max_target_length=dict(
                default=16, type=int, help="Max target tokenized length"
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
        max_total_length: int,
        max_target_length: int,
        tokenizer: PreTrainedTokenizerBase,
        crop_size: Optional[Union[int, Iterable[int]]] = None,
        image_dir: Optional[str] = None,
        logging_level: Optional[Union[int, str]] = None,
        preprocess_on_fetch: bool = False,
        entity_linker_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Init.

        Args:
            See `Twitter201XDataset`.
            max_total_length: max length for targetless tweet and
                target sequence.
            max_target_length: max length for target sequence.
            crop_size: dimension to crop image to, if any.
            preprocess_on_fetch: whether not to apply image transformation
                once when loading the images or every time they are fetched.
                Enable if finetuning the image encoder or using augmentation.
        """

        super().__init__(
            dir,
            kind,
            tokenizer,
            image_dir=image_dir,
            logging_level=logging_level,
            entity_linker_kwargs=entity_linker_kwargs,
            **{
                "max_total_length": max_total_length,
                "max_target_length": max_target_length,
            },
        )

        self.image_transformation = self._get_image_transform(crop_size)
        self.preprocess_on_fetch = preprocess_on_fetch
        if not preprocess_on_fetch:
            self.logger.debug("Preprocessing images")
            self.transform_images()
            self.logger.debug("Done preprocessing images")

    def define_data_struct(self) -> "Twitter201XData":
        """Defines `data` structure."""

        class Twitter201XData(RecordClass):
            """Container of all basic DATA from a Twitter-201{5, 7} example
            for TomBert.

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
            input_mask: List[int]
            type_ids: List[int]
            target_input_ids: List[int]
            target_input_mask: List[int]
            target_type_ids: List[int]
            image: torch.Tensor
            label_id: int

        return Twitter201XData

    def transform_images(self):
        """Preprocesses the images at once."""
        for _id in self.ids:
            self.data[_id].image = self.image_transformation(
                self.data[_id].image
            )

    def encode_plus(
        self,
        examples: List[Twitter201XInfo],
        max_total_length: int,
        max_target_length: int,
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
        max_dataset_tweet_length = 0
        max_dataset_target_length = 0
        max_dataset_total_length = 0

        data = {}

        for example in examples:
            ### update length values
            tless_tweet_tokens = self.tokenizer.tokenize(
                example.targetless_tweet
            )
            target_tokens = self.tokenizer.tokenize(example.target)
            max_dataset_tweet_length = max(
                max_dataset_tweet_length, len(tless_tweet_tokens)
            )
            max_dataset_target_length = max(
                max_dataset_target_length, len(target_tokens)
            )
            max_dataset_total_length = max(
                max_dataset_total_length,
                len(tless_tweet_tokens) + len(target_tokens),
            )
            ###

            target_input = self.tokenizer.encode_plus(
                example.target,
                max_length=max_target_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            # NOTE: this truncates preferably from the target if the two are equal
            input = self.tokenizer.encode_plus(
                example.targetless_tweet,
                example.target,
                max_length=max_total_length,
                truncation="longest_first",  # is already the default
                padding="max_length",
                return_tensors="pt",
            )

            # NOTE: reads same image multiple times
            image, error = self.load_image(example)

            if error:
                err_cnt += 1

            data[int(example.id)] = self.data_class(
                id=int(example.id),
                input_ids=input["input_ids"][0],
                input_mask=input["attention_mask"][0],
                type_ids=input["token_type_ids"][0],
                target_input_ids=target_input["input_ids"][0],
                target_input_mask=target_input["attention_mask"][0],
                target_type_ids=target_input["token_type_ids"][0],
                image=image,
                label_id=torch.tensor(self.label_mapping[example.label]),
            )

        self.logger.info(f"*** Example Data ***\n{next(iter(data.values()))}\n")
        msg = f"{err_cnt} errors occured whilst loading images"
        if err_cnt > 0:
            self.logger.warning(msg)
        else:
            self.logger.info(msg)
        self.logger.info(
            f"Max total length: {max_dataset_total_length}, "
            f"Max tweet length: {max_dataset_tweet_length}, "
            f"Max target length: {max_dataset_target_length}"
        )

        return data

    def replace_images_with_embeddings(
        self, embedding_dict: Dict[int, torch.Tensor]
    ):
        """Replaces the images (or whatever the image attribute of
        the data holds) with the corresponding embeddings.

        Args:
            embedding_dict: the dict that holds the embeddings, indexed
                by the ID of the example.
        """
        if not self.preprocess_on_fetch:
            for _id, embedding in embedding_dict.items():
                self.data[_id].image = embedding

    def __getitem__(self, index: int) -> Tuple[torch.Tensor]:
        """See `Twitter201XData` attribute order to get return order."""
        # NOTE: this returns ID too.

        datum = self.data[self.ids[index]]

        if not self.preprocess_on_fetch:
            return tuple(asdict(datum).values())

        image = self.image_transformation(datum.image)

        return tuple(
            [val if k != "image" else image for k, val in asdict(datum).items()]
        )

    @staticmethod
    def _get_image_transform(
        crop_size: Union[int, Iterable[int]]
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

        # crop_transform = (
        #     transforms.RandomCrop(crop_size)
        #     if crop_size is not None
        #     else transforms.Lambda(lambda x: x)
        # )

        crop_transform = (
            transforms.Compose(
                [
                    transforms.Resize(crop_size),
                    transforms.CenterCrop(crop_size),
                ]
            )
            if crop_size is not None
            else transforms.Lambda(lambda x: x)
        )

        image_transform = transforms.Compose(
            [
                crop_transform,
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )
        return image_transform
