# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Inference of vega model."""
import argparse
import pickle
import vega


def parse_args_parser():
    """Parse parameters."""
    parser = argparse.ArgumentParser(description='Vega Inference.')
    parser.add_argument("--model_desc", default=None, type=str)
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--data_type", default=None, type=str)
    parser.add_argument("--data_path", default=None, type=str)
    parser.add_argument("--backend", default='pytorch', type=str)
    parser.add_argument("--device_category", default='GPU', type=str)
    parser.add_argument("--result_path", default='./result.pkl', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args_parser()
    vega.set_backend(args.backend, args.device_category)
    from vega.model_zoo import ModelZoo
    from vega.datasets.pytorch.common.dataset import Dataset
    dataset = Dataset(type=args.data_type, mode='test', data_path=args.data_path)
    valid_dataloader = dataset.dataloader
    model = ModelZoo.get_model(args.model_desc, args.model)
    result = ModelZoo.infer(model, valid_dataloader)
    output = open(args.result_path, 'wb')
    pickle.dump(result, output)
    output.close()
