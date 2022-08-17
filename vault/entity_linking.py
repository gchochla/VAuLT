import os
import argparse
from numbers import Number
from typing import Union, Tuple, List, Dict, Any

import wikipedia
from REL.ner import load_flair_ner, Cmns
from REL.mention_detection import MentionDetection
from REL.entity_disambiguation import EntityDisambiguation
from REL.utils import process_results
import torch.nn as nn
from transformers import PreTrainedTokenizerBase


class EntityLinker:
    """Links targets from tweets to Wikipedia entities and fetches the
    basic Wikipedia description to be used to create a representation.

    Attributes:
        input_formatter: formats tweets for `EntityDisambiguation`.
        model: entity disambiguation.
        tagger: entity extraction.
        threshold: confidence threshold.
    """

    def __init__(self, root_dir: str, wiki_version: str, threshold: Number):
        """Init.

        Args:
            root_dir: path to downloaded REL files.
            wiki_version: version of wiki to use, e.g. `"wiki_2019"`.
            threshold: confidence threshold.
        """
        self.input_formatter = MentionDetection(root_dir, wiki_version)
        config = dict(
            mode="eval",
            model_path=os.path.join(
                root_dir, wiki_version, "generated", "model"
            ),
        )
        self.model = EntityDisambiguation(root_dir, wiki_version, config)

        self.tagger = Cmns(root_dir, wiki_version, n=5)
        # NOTE: flair tagger results in bug
        # self.tagger = load_flair_ner("ner-fast")

        self.threshold = threshold

    def __call__(
        self,
        example: "memes.dataset_utils.Twitter201XInfo",  # avoid circular imports
    ) -> Tuple[str, Union[str, None], Number]:
        """Disambiguates tweet target and fetches Wikipedia description
        if confidence in assigned entity is high enough.'

        Args:
            example: tweet info.

        Returns:
            The entity `str`, the description if confidence is high enough
            else `None`, and the confidence in the prediction.
        """

        doc_id = "test_doc"

        input_text = {
            doc_id: [
                example.targetless_tweet.replace("$T$", example.target),
                # NOTE: next line results in an empty dataset for some reason
                # [[example.targetless_tweet.find("$"), len(example.target)]],
                [],
            ]
        }
        dataset, _ = self.input_formatter.find_mentions(input_text, self.tagger)
        dataset = {
            doc_id: [
                mention
                for mention in dataset[doc_id]
                # NOTE: won't work properly when target appears multiple times
                if mention["mention"] == example.target
            ]
        }
        preds, _ = self.model.predict(dataset)

        if doc_id in preds:  # if an entity was actually found
            res = process_results(dataset, preds, input_text)[doc_id][0]
            entity, conf = res[3], res[5]
            if conf < self.threshold:
                return entity, None, conf
            description = self.get_entity_description(entity)
            return entity, description, conf

    def get_entity_description(self, entity: str) -> str:
        """Fetches Wikipedia page from entity name.

        Args:
            entity: entity name (as it appears in URL).

        Returns:
            First paragraph from Wikipedia.
        """
        done_disambiguating = False
        while not done_disambiguating:
            try:
                wiki = wikipedia.WikipediaPage(entity)
                done_disambiguating = True
            except wikipedia.DisambiguationError as e:
                entity = str(e).split("\n")[1]

        text = wiki.content
        description = text[: text.find("\n")]
        return description


def integrate_entities_into_model(
    model: nn.Module,
    descriptions: List[str],
    tokenizer: PreTrainedTokenizerBase,
):
    """Integrates new token/entities into model embeddings
    by max-pooling input embeddings of entity description tokens.

    Args:
        model: which model to integrate entities in. Should have
            `resize_token_embeddings`, `get_input_embeddings` and
            `set_input_embeddings` implemented. `get_input_embeddings`
            must return an embeddings' class or a `dict` of them.
        descriptions: descriptions of added entities, in the same
            order as the entities were added.
        tokenizer: tokenizer where token entities where added.
    """

    model.resize_token_embeddings(len(tokenizer))
    embeddings_cls = model.get_input_embeddings()
    if not isinstance(embeddings_cls, dict):
        embeddings_iter = [embeddings_cls]
    else:
        embeddings_iter = embeddings_cls.values()

    for ecls in embeddings_iter:
        embeddings = ecls.weight.clone()
        for i, description in enumerate(reversed(descriptions)):
            desc_ids = tokenizer.encode(description)
            desc_embedding = embeddings[desc_ids].max(0)[0]
            embeddings[-(i + 1)] = desc_embedding
        ecls.weight = nn.parameter.Parameter(embeddings)

    model.set_input_embeddings(embeddings_cls)


def set_entity_linker_subparser(parser: argparse.ArgumentParser):
    """Creates `entity_linker` subparser.

    Args:
        parser: argparser to augment.
    """

    el_subparser = parser.add_subparsers(
        help="multi-argument optional utilities", dest="entity_linker"
    )
    el_parser = el_subparser.add_parser("entity_linker")
    el_parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="root dir of entity linking downloaded files",
    )
    el_parser.add_argument(
        "--wiki_version",
        type=str,
        default="wiki_2019",
        help="which wiki to use",
    )
    el_parser.add_argument(
        "--threshold", type=float, default=10000, help="confidence threshold"
    )


def get_entity_linker_kwargs(
    args: argparse.Namespace,
) -> Union[None, Dict[str, Any]]:
    """Parses args of entity_linking subparser into
    necessary kwargs for datasets, else `None` (default
    for no entity linking).

    Args:
        args: parsed args.
    """
    if args.entity_linker is not None:
        return dict(
            root_dir=args.root_dir,
            wiki_version=args.wiki_version,
            threshold=args.threshold,
        )
