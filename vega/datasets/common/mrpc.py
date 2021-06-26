# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is a class for Bert Tokenizer."""
import logging
import os
import csv
from vega.datasets.common.dataset import Dataset
from vega.common import ClassFactory, ClassType
from ..conf.mrpc import MrpcConfig
from vega.common.config import Config
from pytorch_pretrained_bert import BertTokenizer


@ClassFactory.register(ClassType.DATASET)
class MrpcDataset(Dataset):
    """MRPC data set (GLUE version)."""

    config = MrpcConfig()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        label_list = self.get_labels()

        tokenizer = BertTokenizer.from_pretrained(self.args.vocab_file, do_lower_case=self.args.do_lower_case)
        if tokenizer is None:
            raise ValueError("Tokenizer can't be None.")
        if self.mode == 'train':
            examples = self.get_train_examples(self.args.data_path)
        elif self.mode == 'val':
            examples = self.get_val_examples(self.args.data_path)
        else:
            examples = self.get_test_examples(self.args.data_path)
        self.examples = self.convert_examples_to_features(examples, label_list, self.args.max_seq_length, tokenizer)

    def __getitem__(self, idx):
        """Get item."""
        example = self.examples[idx]
        input_ids = example.get('input_ids')
        input_mask = example.get('input_mask')
        segment_ids = example.get('segment_ids')
        label_ids = example.get('label_id')
        if self.transforms is not None:
            input_ids, input_mask, segment_ids, label_ids = self.transforms(input_ids, input_mask, segment_ids,
                                                                            label_ids)
        target = label_ids
        data = dict(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=label_ids)
        return data, target

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.examples)

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_val_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Create examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = None if set_type == "test" else line[0]
            examples.append(Config(dict(guid=guid, text_a=text_a, text_b=text_b, label=label)))
        return examples

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Read a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

    def convert_examples_to_features(self, examples, label_list, max_seq_length, tokenizer):
        """Load a data file into a list of `InputBatch`s."""
        label_map = {label: i for i, label in enumerate(label_list)}
        features = []
        for (ex_index, example) in enumerate(examples):
            tokens_a = tokenizer.tokenize(example.text_a)

            tokens_b = None
            if example.text_b:
                tokens_b = tokenizer.tokenize(example.text_b)
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0   0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambigiously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            label_id = label_map[example.label]
            if ex_index < 5:
                logging.info("*** Example ***")
                logging.info("guid: %s" % (example.guid))
                logging.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logging.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logging.info("label: %s (id = %d)" % (example.label, label_id))

            features.append(Config(
                dict(input_ids=input_ids,
                     input_mask=input_mask,
                     segment_ids=segment_ids,
                     label_id=label_id)))
        return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncate a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
