import os
import argparse
import logging
from copy import copy

from transformers import AutoTokenizer

from vault.models.tombert import (
    TomBertWithResNetForTMSC,
    TomBertDatasetForTMSC,
    TomBertTrainerForTMSC,
)
from vault.models.tomvilt import TomViltWithResNetForTMSC
from vault.utils import LOGGING_FORMAT
from vault.entity_linking import (
    integrate_entities_into_model,
    set_entity_linker_subparser,
    get_entity_linker_kwargs,
)
from vault.train_utils import MyTrainingArguments
from vault.logging_utils import ExperimentHandler

from utils import general_argparse_args, add_arguments

MODEL = {
    "TomBERT": TomBertWithResNetForTMSC,
    "TomViLT": TomViltWithResNetForTMSC,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--add_placeholder_token",
        action="store_true",
        help="whether to create $T$ token explicitly",
    )

    sp = parser.add_subparsers(dest="model", required=True)

    for model in MODEL:
        sp_model = sp.add_parser(model)
        add_arguments(sp_model, MODEL[model].argparse_args)
        add_arguments(sp_model, TomBertDatasetForTMSC.argparse_args)
        add_arguments(sp_model, TomBertTrainerForTMSC.argparse_args)
        add_arguments(sp_model, general_argparse_args)
        # NOTE: use "entity_linker" to activate
        set_entity_linker_subparser(sp_model)

    return parser.parse_args()


def main():

    args = parse_args()

    reps = args.reps
    del args.reps
    description = args.description
    del args.description

    entity_linker_kwargs = get_entity_linker_kwargs(args)

    logging_level = getattr(logging, args.logging_level)
    logging.basicConfig(
        level=logging_level,
        filename=args.logging_file,
        format=LOGGING_FORMAT,
    )
    del args.logging_level
    del args.logging_file

    if args.model == "TomViLT":
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tweet_model_name_or_path or args.model_name_or_path
        )

    if args.add_placeholder_token:
        tokenizer.add_tokens(["$T$"])

    train_dataset = TomBertDatasetForTMSC(
        dir=args.dir,
        kind=args.train_split,
        max_total_length=args.max_total_length,
        max_target_length=args.max_target_length,
        tokenizer=tokenizer,
        crop_size=args.crop_size,
        preprocess_on_fetch=args.preprocess_on_fetch,
        entity_linker_kwargs=entity_linker_kwargs,
    )
    dev_dataset = (
        TomBertDatasetForTMSC(
            dir=args.dir,
            kind=args.dev_split,
            max_total_length=args.max_total_length,
            max_target_length=args.max_target_length,
            tokenizer=tokenizer,
            crop_size=args.crop_size,
            preprocess_on_fetch=False,
            entity_linker_kwargs=entity_linker_kwargs,
        )
        if args.dev_split is not None
        else None
    )
    test_dataset = (
        TomBertDatasetForTMSC(
            dir=args.dir,
            kind=args.test_split,
            max_total_length=args.max_total_length,
            max_target_length=args.max_target_length,
            tokenizer=tokenizer,
            crop_size=args.crop_size,
            preprocess_on_fetch=False,
            entity_linker_kwargs=entity_linker_kwargs,
        )
        if args.test_split is not None
        else None
    )

    eval_steps = (
        args.eval_steps
        or (len(train_dataset) + args.train_batch_size - 1)
        // args.train_batch_size
    )

    training_args = MyTrainingArguments(
        output_dir=None,
        learning_rate=args.lr,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_ratio=args.warmup_ratio,
        no_cuda=args.device == "cpu",
        disable_tqdm=args.disable_tqdm,
        model_load_filename=args.model_load_filename,
        eval_steps=eval_steps,
        correct_bias=args.correct_bias,
        model_save=args.model_save,
    )

    logging.info(args)

    for rep in range(reps):
        if reps > 1:
            print("\n", f"Rep {rep+1}", "\n")

        experiment_handler = ExperimentHandler(
            "./experiment_logs", args.model + "TMSC", description=description
        )

        args = experiment_handler.set_namespace_params(args)
        training_args = experiment_handler.set_namespace_params(training_args)

        if args.model == "TomViLT":
            model = TomViltWithResNetForTMSC.from_pretrained(
                args.model_name_or_path,
                args.vilt_model_name_or_path,
                resnet_depth=args.resnet_depth,
                n_classes=3,
            )
        else:
            model = TomBertWithResNetForTMSC.from_pretrained(
                args.model_name_or_path,
                args.tweet_model_name_or_path,
                n_classes=3,
                resnet_depth=args.resnet_depth,
                pooling=args.pooling,
                num_hidden_cross_layers=args.num_hidden_cross_layers,
            )

        if args.add_placeholder_token:
            model.resize_token_embeddings(len(tokenizer))

        entity_descriptions = copy(train_dataset.entity_descriptions)
        if dev_dataset is not None:
            entity_descriptions.extend(dev_dataset.entity_descriptions)
        if test_dataset is not None:
            entity_descriptions.extend(test_dataset.entity_descriptions)

        integrate_entities_into_model(model, entity_descriptions, tokenizer)

        # setup parents and disable param for comparison
        experiment_handler.disable_params(["disable_tqdm", "device", "no_cuda"])

        # set up filename
        extra_names = (
            {
                "_model_name_": os.path.split(args.model_name_or_path)[-1],
                "_bert_model_name_": os.path.split(
                    args.tweet_model_name_or_path
                )[-1]
                if args.tweet_model_name_or_path is not None
                else None,
                "_dataset_": train_dataset.name,
            }
            if args.model == "TomBERT"
            else {
                "_model_name_": os.path.split(args.model_name_or_path)[-1],
                "_vilt_": os.path.split(args.vilt_model_name_or_path)[-1],
                "_dataset_": train_dataset.name,
            }
        )
        experiment_handler.set_dict_params(extra_names)
        experiment_handler.disable_params(list(extra_names))
        names_list = list(extra_names) + [
            "add_placeholder_token",
        ]
        if args.entity_linker is not None:
            names_list.append("threshold")
        experiment_handler.name_params(names_list)

        trainer = TomBertTrainerForTMSC(
            model,
            train_dataset,
            experiment_handler,
            dev_dataset=dev_dataset,
            test_dataset=test_dataset,
            logging_level=logging_level,
        )
        trainer.train()


if __name__ == "__main__":
    main()
