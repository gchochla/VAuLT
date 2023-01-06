import os
import argparse
import logging
from copy import copy

from vault.models.vault import (
    VaultForTMSC,
    VaultDatasetForTMSC,
    VaultDatasetForBloombergTwitterCorpus,
    VaultDatasetForMVSA,
    VaultTrainerForTMSC,
    VaultTrainerForBloombergTwitterCorpus,
    VaultTrainerForMVSA,
    VaultProcessor,
)
from vault.utils import LOGGING_FORMAT
from vault.entity_linking import (
    integrate_entities_into_model,
    set_entity_linker_subparser,
    get_entity_linker_kwargs,
)
from vault.logging_utils import ExperimentHandler
from vault.train_utils import MyTrainingArguments
from vault.utils import twitter_preprocessor
from vault.utils import demojizer_selector


from utils import general_argparse_args, add_arguments


def num_outputs(task, preprocessed=False):
    if task == "Twitter201X":
        return 3
    if task == "MVSA":
        return 3 * (int(not preprocessed) + 1)
    if task == "Bloomberg":
        return 1


DATASET = {
    "Twitter201X": VaultDatasetForTMSC,
    "Bloomberg": VaultDatasetForBloombergTwitterCorpus,
    "MVSA": VaultDatasetForMVSA,
}
TRAINER = {
    "Twitter201X": VaultTrainerForTMSC,
    "Bloomberg": VaultTrainerForBloombergTwitterCorpus,
    "MVSA": VaultTrainerForMVSA,
}


def parse_args():
    parser = argparse.ArgumentParser()
    sp = parser.add_subparsers(dest="task", required=True)

    for task in DATASET:
        sp_task = sp.add_parser(task)
        add_arguments(sp_task, VaultForTMSC.argparse_args)
        add_arguments(sp_task, DATASET[task].argparse_args)
        add_arguments(sp_task, TRAINER[task].argparse_args)
        add_arguments(sp_task, general_argparse_args)
        if task == "Twitter201X":
            sp_task.add_argument(
                "--add_placeholder_token",
                action="store_true",
                help="whether to create $T$ token explicitly",
            )
            set_entity_linker_subparser(sp_task)

    return parser.parse_args()


def main():

    args = parse_args()

    task = args.task
    del args.task

    reps = args.reps
    del args.reps
    description = args.description
    del args.description
    max_num_workers = args.max_num_workers
    del args.max_num_workers

    logging_level = getattr(logging, args.logging_level)
    logging.basicConfig(
        level=logging_level,
        filename=args.logging_file,
        format=LOGGING_FORMAT,
    )
    del args.logging_level
    del args.logging_file

    processor = VaultProcessor.from_pretrained(
        args.vilt_model_name_or_path, args.bert_model_name_or_path
    )

    if hasattr(args, "add_placeholder_token") and args.add_placeholder_token:
        processor.tokenizer.add_tokens(["$T$"])

    task_dataset_kwargs = (
        dict(
            dir=args.dir,
            tokenizer=processor,
            crop_size=args.crop_size,
            preprocess_on_fetch=args.preprocess_on_fetch,
            entity_linker_kwargs=get_entity_linker_kwargs(args),
        )
        if task == "Twitter201X"
        else dict(
            root_dir=args.root_dir,
            processor=processor,
            twitter_preprocessor=twitter_preprocessor(),
            demojizer=demojizer_selector(args.bert_model_name_or_path),
            image_augmentation=args.image_augmentation,
        )
        if task == "Bloomberg"
        else dict(
            root_dir=args.root_dir,
            processor=processor,
            twitter_preprocessor=twitter_preprocessor(),
            demojizer=demojizer_selector(args.bert_model_name_or_path),
            image_augmentation=args.image_augmentation,
            preprocessed=args.preprocessed,
        )
    )
    task_split_key = "kind" if task == "Twitter201X" else "splits"

    train_dataset = DATASET[task](
        max_length=args.max_length,
        **{task_split_key: args.train_split, **task_dataset_kwargs},
    )
    dev_split = args.dev_split if hasattr(args, "dev_split") else args.val_split
    dev_dataset = (
        DATASET[task](
            max_length=args.max_length,
            **{
                task_split_key: dev_split,
                **task_dataset_kwargs,
            },
        )
        if dev_split is not None
        else None
    )
    test_dataset = (
        DATASET[task](
            max_length=args.max_length,
            **{task_split_key: args.test_split, **task_dataset_kwargs},
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
        dataloader_num_workers=min(max_num_workers, args.train_batch_size),
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
            "./experiment_logs", f"VaultTMSC{task}", description=description
        )

        args = experiment_handler.set_namespace_params(args)
        training_args = experiment_handler.set_namespace_params(training_args)

        n_classes = (
            num_outputs(task)
            if task != "MVSA"
            else num_outputs(task, args.preprocessed)
        )

        model = VaultForTMSC.from_pretrained(
            args.vilt_model_name_or_path,
            args.bert_model_name_or_path,
            freeze_lm=args.freeze_lm,
            n_classes=n_classes,
            vilt_dropout_prob=args.vilt_dropout_prob,
        )

        if (
            hasattr(args, "add_placeholder_token")
            and args.add_placeholder_token
        ):
            model.resize_token_embeddings(len(processor.tokenizer))

        if task == "Twitter201X":
            entity_descriptions = copy(train_dataset.entity_descriptions)
            if dev_dataset is not None:
                entity_descriptions.extend(dev_dataset.entity_descriptions)
            if test_dataset is not None:
                entity_descriptions.extend(test_dataset.entity_descriptions)

            integrate_entities_into_model(
                model, entity_descriptions, processor.tokenizer
            )

        # setup parents and disable param for comparison
        experiment_handler.disable_params(["disable_tqdm", "device", "no_cuda"])

        # set up filename
        extra_names = {
            "_model_name_": os.path.split(args.vilt_model_name_or_path)[-1],
            "_bert_model_name_": os.path.split(args.bert_model_name_or_path)[-1]
            if args.bert_model_name_or_path is not None
            else None,
            "_dataset_": train_dataset.name,
        }
        experiment_handler.set_dict_params(extra_names)
        experiment_handler.disable_params(list(extra_names))
        names_list = list(extra_names)
        if task == "Twitter201X":
            names_list += [
                "add_placeholder_token",
            ]
            if args.entity_linker is not None:
                names_list.append("threshold")
        experiment_handler.name_params(names_list)

        trainer = TRAINER[task](
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
