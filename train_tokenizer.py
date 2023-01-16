from dataclasses import dataclass, field
from typing import Optional

import datasets
from transformers import AutoTokenizer
from transformers import HfArgumentParser


def batch_iterator(
    dataset,
    batch_size: int=1_000,
    max_samples: Optional[int]=None,
):
    count = 0
    for batch in dataset.iter(batch_size=batch_size):
        yield batch["text"]
        count += len(batch["text"])
        if max_samples is not None and count > max_samples:
            break


@dataclass
class TokenizerTrainingArguments:
    """
    Configuration for tokenizer training.
    """

    base_tokenizer: Optional[str] = field(
        default="gpt2",
        metadata={"help": "Base tokenizer to build new tokenizer from."}
    )
    dataset_id: str = field(
        default="gabrielaltay/pmc-open-access",
        metadata={"help": "Dataset name or path."}
    )
    text_column: Optional[str] = field(
        default="text",
        metadata={"help": "Column containing text data to process."}
    )
    vocab_size: Optional[int] = field(
        default=32_768,
        metadata={"help": "Number of examples to train tokenizer on."}
    )
    max_examples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of samples to train the tokenizer on."}
    )
    tokenizer_name: Optional[str] = field(
        default="pmc-open-access",
        metadata={"help": "Name of new tokenizer."}
    )
    push_to_hub: Optional[bool] = field(
        default=False,
        metadata={"help": "Push saved tokenizer to the hub."}
    )
    ignore_verifications: Optional[bool] = field(
        default=True,
        metadata={"help": "If True, skip computing checksums of downloaded files."}
    )
    num_proc: Optional[int] = field(
        default=1,
        metadata={"help": "Number of processes to use in load_dataset."}
    )

if __name__ == "__main__":

    parser = HfArgumentParser(TokenizerTrainingArguments)
    args = parser.parse_args()

    base_tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer)
    ds_train = datasets.load_dataset(
        args.dataset_id,
        split="train",
        ignore_verifications=args.ignore_verifications,
        num_proc=args.num_proc,
    )
    ds_train = ds_train.remove_columns([
        col for col in ds_train.column_names if col != args.text_column
    ])
    tokenizer = base_tokenizer.train_new_from_iterator(
        batch_iterator(ds_train),
        vocab_size=args.vocab_size,
    )
    tokenizer.save_pretrained(args.tokenizer_name, push_to_hub=args.push_to_hub)
