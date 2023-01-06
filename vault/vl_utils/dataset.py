import os
import logging
import random
from copy import copy, deepcopy
from abc import ABC, abstractmethod
from PIL import Image, ImageFile
from collections import Counter
from typing import Any, List, Tuple, Optional, Dict, Union, Iterable, Callable

import torch
import pandas as pd
from transformers import ViltProcessor
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from vault.utils import flatten_list
from .dataset_utils import normalize_word, safe_dict_concat

ImageFile.LOAD_TRUNCATED_IMAGES = True


class VisionAndLanguageDataset(Dataset, ABC):
    """Pytorch base dataset for Vision and Language tasks. Dataset
    implements lazy loading of images.

    Attributes:
        logger: logging module.
        root_dir: root directory of dataset.
        splits: which split(s) to use.
        name: dataset name.
        processor: text and image "tokenizer".
        ids: identifiers of examples.
        texts: list of sentences per example.
        image_fns: corresponding image filenames.
        labels: corresponding labels.
        inputs: if not lazy, the tokenized inputs.
        images: if not lazy, the images.
        effective_inds: because we may have multiple texts per image
            or vice versa, we have tuples in our data structure. To
            allow the user to use simple integers, we map them to
            tuple indices.
        image_preprocessor: image preprocessing.
        encode_kwargs: kwargs for the tokenization process.
        _is_train: if train split is included.
    """

    argparse_args = dict(
        root_dir=dict(required=True, type=str, help="dataset root directory"),
        train_split=dict(
            default="train",
            type=str,
            nargs="+",
            help="train split(s)",
        ),
        val_split=dict(
            type=str,
            nargs="+",
            help="development split(s)",
        ),
        test_split=dict(
            type=str,
            nargs="+",
            help="test split(s)",
        ),
        crop_size=dict(
            default=224, type=int, help="dimension to crop images to"
        ),
        image_augmentation=dict(
            action="store_true", help="whether to use image augmentation"
        ),
    )

    def __init__(
        self,
        root_dir: Union[List[str], str],
        splits: Union[List[str], str],
        processor: ViltProcessor,
        encode_kwargs: Dict[str, Any],
        crop_size: Union[int, Tuple[int]],
        logging_level: Optional[int] = None,
        lazy: bool = True,
        image_augmentation: bool = False,
    ):
        """Init.

        Args:
            root_dir: root directory of dataset.
            splits: which split(s) to use.
            processor: text and image "tokenizer".
            encode_kwargs: kwargs for the tokenization process.
            crop_size: crop size input.
            logging_level: level to log at.
            lazy: whether to load an image when it is requested (every time)
                or pre-load them all.
        """

        self.logger = logging.getLogger(__name__)
        if not logging_level:
            logging_level = logging.WARNING
        self.logger.setLevel(logging_level)

        self.root_dir = root_dir
        self.splits = splits

        if not isinstance(self.splits, list):
            self.splits = [self.splits]

        self._is_train = "train" in self.splits

        self.name = (
            os.path.basename(os.path.abspath(self.root_dir))
            + "("
            + ",".join(self.splits)
            + ")"
        )

        self.processor = processor

        (
            self.ids,
            self.texts,
            self.image_fns,
            self.labels,
            *kwargs,
        ) = self.load_dataset()

        if isinstance(self.texts[0], str):
            self.texts = [[text] for text in self.texts]

        try:
            for k, v in kwargs[0].items():
                setattr(self, k, v)
        except:
            pass

        # indices we actually have to use to grab data
        self.effective_inds = [
            (i, j)
            for i, example_texts in enumerate(self.texts)
            for j in range(len(example_texts))
        ]

        self.image_preprocessor = self._init_image_preprocessor(crop_size)
        self.image_transformation = self._init_image_transformation(crop_size)

        self.encode_kwargs = encode_kwargs

        if not lazy:
            self.images = self._load_images()
            image = [self.get_image(*i) for i in self.effective_inds]
            text = [self.get_text(*i) for i in self.effective_inds]
            if not image_augmentation:
                self.inputs = self.encode_plus(image=image, text=text)
            else:
                self.inputs = None
        else:
            self.images = None
            self.inputs = None

    @property
    def text_tokenizer(self):
        """Defines the text tokenizer based
        on the multimodal data (pre)processor."""
        return self.processor.tokenizer

    def __len__(self):
        return len(self.effective_inds)

    def text_preprocessor(self, text: str) -> str:
        """Returns the preprocessed text"""
        return text

    def encode_plus(
        self,
        image: Union[torch.Tensor, Iterable[torch.Tensor]],
        text: Union[str, Iterable[str]],
        batch_size: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Prepares the inputs of the multimodal models
        (`transformers` style).

        Args:
            image: tensor image or list of them.
            text: text or list of text. Must correspond to `image`.

        Returns:
            `transformers`-style input data.
        """

        if batch_size is None:
            return self.processor(
                images=image,
                text=text,
                padding="max_length",
                max_length=self.encode_kwargs["max_length"],
                truncation=self.encode_kwargs.get(
                    "truncation", "longest_first"
                ),
                return_tensors="pt",
            )

        inputs = None
        for i in range((len(image) + batch_size - 1) // batch_size):
            concat_dict = self.processor(
                images=image[i * batch_size : (i + 1) * batch_size],
                text=text[i * batch_size : (i + 1) * batch_size],
                padding="max_length",
                max_length=self.encode_kwargs["max_length"],
                truncation=self.encode_kwargs.get(
                    "truncation", "longest_first"
                ),
                return_tensors="pt",
            )

            if inputs is not None:
                inputs = safe_dict_concat([inputs, concat_dict])
            else:
                inputs = concat_dict

        return inputs

    @abstractmethod
    def load_dataset(
        self,
    ) -> Tuple[
        List[str],
        Union[List[str], List[List[str]]],
        Union[List[str], List[List[str]]],
        torch.Tensor,
    ]:
        """Loads dataset. Returns IDs, texts, image filenames and labels."""

    def _get_image_from_fn(self, img_fn: str) -> torch.Tensor:
        """Loads and transforms image from filename.

        Args:
            img_fn: image filename.

        Returns:
            Transformed image.
        """
        # try:
        image = Image.open(img_fn).convert("RGB")
        return self.image_preprocessor(image)
        # except:
        #     # TODO: handle missing images
        #     pass

    def _load_images(self):
        """Loads all images at once."""
        images = [self.get_image(i, None) for i in range(len(self.texts))]
        return images

    def get_image(self, eff_index: int, sub_index: int) -> torch.Tensor:
        """Loads and transforms image.

        Args:
            eff_index: effective text index.
            sub_index: index within the same batch of texts.

        Returns:
            Transformed image.
        """

        if hasattr(self, "images") and self.images is not None:
            return self.images[eff_index]

        return self._get_image_from_fn(self.image_fns[eff_index])

    def get_text(self, eff_index: int, sub_index: int) -> str:
        return self.text_preprocessor(self.texts[eff_index][sub_index])

    def get_label(self, eff_index: int, sub_index: int) -> torch.Tensor:
        return self.labels[eff_index]

    def __getitem__(
        self, index: int
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Returns a single example in the `transformers` format
        and its label(s)."""
        eff_index, sub_index = self.effective_inds[index]
        if self.inputs is None:
            text = self.get_text(eff_index, sub_index)
            image = self.get_image(eff_index, sub_index)
            if "train" in self.splits:
                image = self.image_transformation(image)
            inputs = self.encode_plus(image, text)
        else:
            inputs = {k: v[index] for k, v in self.inputs.items()}

        label = self.get_label(eff_index, sub_index)

        return inputs, label

    def collate_fn(self, batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Returns multiple examples in the `transformers` format
        and their labels."""
        data, labels = [b[0] for b in batch], [b[1] for b in batch]
        return safe_dict_concat(data), torch.stack(labels)

    def _init_image_preprocessor(self, crop_size):
        """Returns image transformation for image preprocessing."""
        return transforms.Lambda(lambda x: x)

    def _init_image_transformation(self, crop_size):
        """Returns image transformation used for augmentation, etc."""
        return transforms.Lambda(lambda x: x)


class BloombergTwitterCorpus(VisionAndLanguageDataset):
    """Base class for Bloomberg Twitter text-image dataset.

    Attributes:
        See `VisionAndLanguageDataset`.
        tasks: task(s) to consider.
        label_names: names of labels.
        task_inds: indices of chosen tasks in labels.
        twitter_preprocessor: twitter-specific text preprocessor.
        demojizer: transforms emojis to text.
    """

    _dev_size = 564
    _test_size = 704

    def __init__(
        self,
        root_dir: Union[List[str], str],
        splits: Union[List[str], str],
        processor: ViltProcessor,
        encode_kwargs: Dict[str, Any],
        crop_size: Union[int, Tuple[int]],
        tasks: Union[str, List[str]] = "text_is_represented",
        twitter_preprocessor: Optional[Callable] = None,
        demojizer: Optional[Callable] = None,
        image_augmentation: bool = False,
        _dev_size: Optional[int] = None,
        _test_size: Optional[int] = None,
        logging_level: Optional[int] = None,
    ):
        """Init.

        Args:
            Check `VisionAndLanguageDataset` except for `lazy`,
                which is set to `False`.
            tasks: which task(s) from the ones available should be considered.
            twitter_preprocessor: twitter-specific text preprocessor.
            demojizer: transforms emojis to text.
        """

        self._dev_size = _dev_size or self._dev_size
        self._test_size = _test_size or self._test_size

        self.tasks = tasks
        if isinstance(self.tasks, str):
            self.tasks = [self.tasks]
        self.twitter_preprocessor = twitter_preprocessor or (lambda x: x)
        self.demojizer = demojizer or (lambda x: x)

        super().__init__(
            root_dir=root_dir,
            splits=splits,
            processor=processor,
            encode_kwargs=encode_kwargs,
            crop_size=crop_size,
            logging_level=logging_level,
            lazy=False,
            image_augmentation=image_augmentation,
        )

        self.task_inds = [self.label_names.index(task) for task in self.tasks]

    def text_preprocessor(self, text: str) -> str:
        """Twitter-specific preprocessor."""
        return self.twitter_preprocessor(self.demojizer(text))

    def load_dataset(
        self,
    ) -> Tuple[
        List[str],
        Union[List[str], List[List[str]]],
        Union[List[str], List[List[str]]],
        torch.Tensor,
        Dict[str, Any],
    ]:
        """Loads dataset. Assumes comma-separated values,
        with \ being the escape character.

        Returns:
            IDs, texts, image filenames, labels and the name of the labels.
        """

        df = pd.read_csv(
            os.path.join(self.root_dir, "bloomberg-textimage.csv"),
            escapechar="\\",
        )
        ids = df.tweet_id.values.tolist()
        texts = df.tweet.values.tolist()
        label_names = df.columns[3:].values.tolist()
        labels = torch.tensor(df.iloc[:, 3:].values, dtype=float)

        image_dir = os.path.join(self.root_dir, "Twitter_images")
        image_fns = [os.path.join(image_dir, f"T{_id}.jpg") for _id in ids]

        # SPLITS
        random.seed(42)
        # not many tweets per ID, naive is fine
        eval_splits_inds = random.sample(
            range(len(ids)), self._dev_size + self._test_size
        )
        train_split_inds = list(
            set(range(len(ids))).difference(eval_splits_inds)
        )
        dev_split_inds = eval_splits_inds[: self._dev_size]
        test_split_inds = eval_splits_inds[self._dev_size :]

        split_inds = (
            (train_split_inds if "train" in self.splits else [])
            + (dev_split_inds if "dev" in self.splits else [])
            + (test_split_inds if "test" in self.splits else [])
        )

        ids = [ids[i] for i in split_inds]
        texts = [texts[i] for i in split_inds]
        image_fns = [image_fns[i] for i in split_inds]
        labels = labels[split_inds]
        # end SPLITS

        return ids, texts, image_fns, labels, dict(label_names=label_names)

    def get_label(self, eff_index: int, sub_index: int) -> torch.Tensor:
        return self.labels[eff_index][self.task_inds].squeeze()


class MVSA(VisionAndLanguageDataset):
    """MVSA (with single or multiple annotators) base class."""

    argparse_args = deepcopy(VisionAndLanguageDataset.argparse_args)
    argparse_args.update(
        dict(
            preprocessed=dict(
                action="store_true",
                help="whether to preprocess the dataset labels like the literature "
                "(e.g. https://dl.acm.org/doi/pdf/10.1145/3132847.3133142)",
            )
        )
    )

    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9736584: 8:1:1 splits
    #   + extra preprocessing for "inconsistent pairs" (remove opposites, if one neutral
    #   and on polarity, keep polarity -> one label per pair).
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9246699: 8:1:1 splits, also preprocessed
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9718029: 8:0.5:1.5 splits, also preprocessed

    _dev_ratio = 0.1
    _test_ratio = 0.1

    def __init__(
        self,
        root_dir: Union[List[str], str],
        splits: Union[List[str], str],
        processor: ViltProcessor,
        encode_kwargs: Dict[str, Any],
        crop_size: Union[int, Tuple[int]],
        twitter_preprocessor: Optional[Callable] = None,
        demojizer: Optional[Callable] = None,
        logging_level: Optional[int] = None,
        preprocessed: bool = True,
        image_augmentation: bool = False,
    ):
        """Init.

        Args:
            Check `VisionAndLanguageDataset` except for `lazy`,
                which is set to `False`.
            twitter_preprocessor: twitter-specific text preprocessor.
            demojizer: transforms emojis to text.
            preprocessed: remove "inconsistent" label pairs, transform
                label pairs in single labels.
        """

        self.twitter_preprocessor = twitter_preprocessor or (lambda x: x)
        self.demojizer = demojizer or (lambda x: x)
        self.preprocessed = preprocessed

        super().__init__(
            root_dir=root_dir,
            splits=splits,
            processor=processor,
            encode_kwargs=encode_kwargs,
            crop_size=crop_size,
            logging_level=logging_level,
            lazy="single" not in root_dir.lower(),
            image_augmentation=image_augmentation,
        )

    def text_preprocessor(self, text: str) -> str:
        """Twitter-specific preprocessor."""
        return self.twitter_preprocessor(self.demojizer(text))

    def load_dataset(self):
        def aggregate_annotators(annotations):
            c = Counter(annotations)
            n_majority = next(
                iter(
                    {
                        k: v
                        for k, v in sorted(
                            c.items(), key=lambda item: item[1], reverse=True
                        )
                    }
                )
            )
            if c[n_majority] >= (len(annotations) + 1) // 2:
                return n_majority
            return

        def aggregate_modalities(annotations, label_mapper):
            if (
                label_mapper["positive"] in annotations
                and label_mapper["negative"] in annotations
            ):
                return
            if label_mapper["positive"] in annotations:
                return label_mapper["positive"]
            elif label_mapper["negative"] in annotations:
                return label_mapper["negative"]
            return label_mapper["neutral"]

        def remove_nones(labels, ids):
            aggr_inds = [
                i
                for i, ls in enumerate(labels)
                if (
                    any(l is None for l in ls)
                    if isinstance(ls, list)
                    else ls is None
                )
            ]
            print(f"Removing {len(aggr_inds)} from {len(ids)}")
            for i in reversed(aggr_inds):
                labels.pop(i)
                ids.pop(i)

        df = pd.read_csv(
            os.path.join(self.root_dir, "labelResultAll.txt"), sep="\t"
        )
        ids = df.ID.values.tolist()

        try:
            with open(os.path.join(self.root_dir, "corrupt_ids.txt")) as fp:
                corrupt_ids = [int(i) for i in fp.readlines()]
            corrupt_inds = [ids.index(_id) for _id in corrupt_ids]

            ids = [_id for i, _id in enumerate(ids) if i not in corrupt_inds]
        except:
            corrupt_inds = []

        str2int = dict(positive=0, neutral=1, negative=2)

        if "text,image.1" in df:  # multiple annotators
            labels = [
                [[str2int[s] for s in li.split(",")] for li in l]
                for i, l in enumerate(
                    zip(
                        df["text,image"].values.tolist(),
                        df["text,image.1"].values.tolist(),
                        df["text,image.2"].values.tolist(),
                    )
                )
                if i not in corrupt_inds
            ]
            labels = [
                [
                    aggregate_annotators([pair[i] for pair in l])
                    for i in range(len(l[0]))
                ]
                for l in labels
            ]

            remove_nones(labels, ids)

        else:
            labels = [
                [str2int[s] for s in l.split(",")]
                for i, l in enumerate(df["text,image"].values.tolist())
                if i not in corrupt_inds
            ]

        if self.preprocessed:
            labels = [aggregate_modalities(l, str2int) for l in labels]
            remove_nones(labels, ids)

        labels = torch.tensor(labels, dtype=int)

        # SPLITS
        _dev_size = max(1, int(self._dev_ratio * len(ids)))
        _test_size = max(1, int(self._test_ratio * len(ids)))
        random.seed(42)
        eval_splits_inds = random.sample(
            range(len(ids)), _dev_size + _test_size
        )
        train_split_inds = list(
            set(range(len(ids))).difference(eval_splits_inds)
        )
        dev_split_inds = eval_splits_inds[:_dev_size]
        test_split_inds = eval_splits_inds[_dev_size:]

        split_inds = (
            (train_split_inds if "train" in self.splits else [])
            + (dev_split_inds if "dev" in self.splits else [])
            + (test_split_inds if "test" in self.splits else [])
        )

        ids = [ids[i] for i in split_inds]
        labels = labels[split_inds]
        # end SPLITS

        texts = []
        image_fns = []

        for _id in ids:
            with open(
                os.path.join(self.root_dir, "data", f"{_id}.txt"),
                encoding="latin1",
            ) as fp:
                texts.append(" ".join(fp.readlines()))
            image_fns.append(os.path.join(self.root_dir, "data", f"{_id}.jpg"))

        return ids, texts, image_fns, labels

    def collate_fn(self, batch) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Returns multiple examples in the `transformers` format
        and their labels."""
        data, labels = [b[0] for b in batch], [b[1] for b in batch]
        return safe_dict_concat(data), torch.stack(labels)
