# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inference of vega detection model."""

import vega
from vega.common import argment_parser
from vega.common import FileOps
from vega.common.general import General
from vega import security


def _load_data(args):
    """Load data from path."""
    if args.data_format == 'CULANE':
        return vega.get_dataset("AutoLaneDataset", dataset_format="CULane", data_path=args.data_path, mode="test",
                                batch_size=args.batch_size).loader
    elif args.data_format == 'COCO':
        return vega.get_dataset("CocoDataset", data_root=args.data_path, mode="test",
                                batch_size=args.batch_size).loader


def _get_model(args):
    """Get model."""
    from vega.model_zoo import ModelZoo
    model = ModelZoo.get_model(args.model_desc, args.model)
    if vega.is_torch_backend():
        if args.device == "GPU":
            model = model.cuda()
        model.eval()
    return model


def _infer(args, loader, model=None):
    """Choose backend."""
    if vega.is_torch_backend():
        return _infer_pytorch(args, model, loader)
    elif vega.is_tf_backend():
        return _infer_tf(args, model, loader)
    elif vega.is_ms_backend():
        return _infer_ms(args, model, loader)


def _infer_pytorch(args, model, loader):
    """Infer with pytorch."""
    infer_result = []
    import torch
    with torch.no_grad():
        for batch in loader:
            if args.data_format == 'CULANE':
                image = batch.pop('image').cuda(non_blocking=True).float()
                infer_result = model(input=image,
                                     forward_switch='valid',
                                     **batch)
            elif args.data_format == 'COCO':
                infer_result = model(**batch)
        return infer_result


def _infer_tf(args, model, loader):
    """Infer with tf."""
    raise ValueError('Not currently supported.')


def _infer_ms():
    """Infer with ms."""
    raise ValueError('Not currently supported.')


def _save_result(args, result):
    """Save results."""
    _output_file = args.output_file
    if not _output_file:
        _output_file = "./result.pkl"
    FileOps.dump_pickle(result, _output_file)
    print('Results of Inference is saved in {}.'.format(_output_file))


def parse_args_parser():
    """Parse parameters."""
    parser = argment_parser('Vega Inference.')
    parser.add_argument("-c", "--model_desc", default=None, type=str, required=True,
                        help="model description file, generally in json format, contains 'module' node.")
    parser.add_argument("-m", "--model", default=None, type=str, required=True,
                        help="model weight file, usually ends with pth, ckpl, etc.")
    parser.add_argument("-df", "--data_format", default="CULANE", type=str, required=True,
                        choices=["COCO",
                                 "CULANE"
                                 ],
                        help="data type, "
                        )
    parser.add_argument("-bs", "--batch_size", default=1, type=str,
                        help="Batch size of inference, default: 1.")
    parser.add_argument("-dp", "--data_path", default=None, type=str, required=True,
                        help="the folder where the file to be inferred is located.")
    parser.add_argument("-b", "--backend", default="pytorch", type=str,
                        choices=["pytorch", "tensorflow", "mindspore"],
                        help="set training platform")
    parser.add_argument("-d", "--device", default="GPU", type=str,
                        choices=["CPU", "GPU", "NPU"],
                        help="set training device")
    parser.add_argument("-o", "--output_file", default=None, type=str,
                        help="output file. "
                        "type: pkl"
                        )
    parser = security.args.add_args(parser)
    args = parser.parse_args()
    security.args.check_args(args)
    return args


def main():
    """Inference."""
    args = parse_args_parser()
    if args.security:
        if not security.load_config("client"):
            print("If you run vega in normal mode, use parameter '-s'.")
            print("For more parameters: vega-inference-det --help")
            return
    General.security = args.security
    vega.set_backend(args.backend, args.device)
    print("Start building model.")
    model = _get_model(args)
    print("Start loading data.")
    loader = _load_data(args)
    print("Start inferencing.")
    result = _infer(args, loader, model)
    _save_result(args, result)
    print("Completed successfully.")


if __name__ == '__main__':
    main()
