# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is a class for Glue dataset."""
import logging
import json
import numpy as np
from collections import namedtuple
from tqdm import tqdm, trange
from vega.datasets.common.dataset import Dataset
from pytorch_pretrained_bert import BertTokenizer
from vega.common.class_factory import ClassType, ClassFactory
from pathlib import Path
from .utils.data_processor import processors, output_modes
from ..conf.glue import GlueConfig

InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids label_id seq_length is_next")


@ClassFactory.register(ClassType.DATASET)
class GlueDataset(Dataset):
    """Glue Dataset."""

    config = GlueConfig()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tokenizer = BertTokenizer.from_pretrained(self.args.vocab_file, do_lower_case=self.args.do_lower_case)
        if self.args.pregenerated:
            self.features = read_features_from_file(tokenizer, self.args.data_path)
        else:
            processor = processors[self.args.task_name]()
            train_examples = processor.get_examples(self.mode, self.args.data_path)
            label_list = processor.get_labels()
            output_mode = output_modes[self.args.task_name]
            self.features = convert_examples_to_features(train_examples, label_list, self.args.max_seq_length,
                                                         tokenizer, output_mode)

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.features)

    def __getitem__(self, item):
        """Get an item of the dataset according to the index."""
        feature = self.features[item]
        data = dict(input_ids=feature.input_ids, attention_mask=feature.input_mask, token_type_ids=feature.segment_ids)
        labels = feature.label_id
        # next_sentence_label=int(self.is_nexts[item]))
        return data, labels


def read_features_from_file(tokenizer, data_path):
    """Read features from file."""
    logging.info('data_path: {}'.format(data_path))
    data_path = Path(data_path)
    data_file = data_path / "epoch_0.json"
    metrics_file = data_path / "epoch_0_metrics.json"
    logging.info('data_file: {}'.format(data_file))
    logging.info('metrics_file: {}'.format(metrics_file))
    assert data_file.is_file() and metrics_file.is_file()
    metrics = json.loads(metrics_file.read_text())
    num_samples = metrics['num_training_examples']
    seq_len = metrics['max_seq_len']
    features = []
    with data_file.open() as f:
        for i, line in enumerate(tqdm(f, total=num_samples, desc="Training examples")):
            line = line.strip()
            example = json.loads(line)
            feature = convert_example_to_features(example, tokenizer, seq_len)
            features.append(feature)
    logging.info("Loading complete!")
    return features


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_mode):
    """Load a data file into a list of `InputBatch`s."""
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        seq_length = len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 1:
            logging.info("*** Example ***")
            logging.info("guid: %s" % (example.guid))
            logging.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logging.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logging.info("label: {}".format(example.label))
            logging.info("label_id: {}".format(label_id))

        features.append(
            InputFeatures(input_ids=np.array(input_ids),
                          input_mask=np.array(input_mask),
                          segment_ids=np.array(segment_ids),
                          label_id=np.array(label_id),
                          seq_length=np.array(seq_length),
                          is_next=None))
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


def convert_example_to_features(example, tokenizer, max_seq_length):
    """Convert example."""
    tokens = example["tokens"]
    segment_ids = example["segment_ids"]
    is_random_next = example["is_random_next"]
    masked_lm_positions = example["masked_lm_positions"]
    masked_lm_labels = example["masked_lm_labels"]

    if len(tokens) > max_seq_length:
        logging.info('len(tokens): {}'.format(len(tokens)))
        logging.info('tokens: {}'.format(tokens))
        tokens = tokens[:max_seq_length]

    if len(tokens) != len(segment_ids):
        logging.info('tokens: {}\nsegment_ids: {}'.format(tokens, segment_ids))
        segment_ids = [0] * len(tokens)

    assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

    input_array = np.zeros(max_seq_length, dtype=np.int64)
    input_array[:len(input_ids)] = input_ids

    mask_array = np.zeros(max_seq_length, dtype=np.int64)
    mask_array[:len(input_ids)] = 1

    segment_array = np.zeros(max_seq_length, dtype=np.int64)
    segment_array[:len(segment_ids)] = segment_ids

    lm_label_array = np.full(max_seq_length, dtype=np.int64, fill_value=-1)
    lm_label_array[masked_lm_positions] = masked_label_ids

    features = InputFeatures(input_ids=input_array,
                             input_mask=mask_array,
                             segment_ids=segment_array,
                             label_id=lm_label_array,
                             seq_length=None,
                             is_next=is_random_next)
    return features
