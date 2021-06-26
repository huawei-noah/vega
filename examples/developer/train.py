# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""The example of training model."""

import vega


# ===================== backend =========================

# set backend, {pytorch, tensorflow, mindspore}, {GPU, NPU}
vega.set_backend("pytorch", "GPU")


# ===================== model =========================

# using vega's model directly
# vega's network can work on pytorch, tensorflwo or mindspore
model = vega.network("ResNet", depth=18).cuda()

# # or using vega's model zoo with model desc
# desc = {
#     "modules": ["backbone"],
#     "backbone": {
#         "type": "ResNet",
#         "depth": 18,
#         "num_class": 10,
#     }
# }
# model = ModelZoo.get_model(model_desc=desc).cuda()

# # or using torchvision model
# from torchvision.models import resnet18, resnet34
# model = resnet18().cuda()


# ===================== dataset =========================

# using vega's dataset, vega's dataset can work on pytorch, tensorflwo or mindspore
train_loader = vega.dataset("Cifar10", data_path="/cache/datasets/cifar10", mode="train", batch_size=256).loader
test_loader = vega.dataset("Cifar10", data_path="/cache/datasets/cifar10", mode="val", batch_size=256).loader

# # or using torchvision dataset
# import torchvision
# import torchvision.transforms as transforms
# train_dataset = torchvision.datasets.CIFAR10(
#     root="/cache/datasets/cifar10", train=True, transform=transforms.ToTensor())
# test_dataset = torchvision.datasets.CIFAR10(
#     root="/cache/datasets/cifar10", train=False, transform=transforms.ToTensor())
# train_loader = torch.utils.data.DataLoader(
#     dataset=train_dataset, batch_size=256, shuffle=True)
# test_loader = torch.utils.data.DataLoader(
#     dataset=test_dataset, batch_size=256, shuffle=False)


# ===================== trainer =========================

trainer = vega.trainer(model=model)
trainer.config.epochs = 2
trainer.config.mixup = True
trainer.train_loader = train_loader
trainer.valid_loader = test_loader
trainer.train_process()
print("Training is complete. Please check folder: {}".format(trainer.get_local_worker_path()))


# ===================== cluster =========================

# from vega.core.scheduler import create_master, shutdown_cluster
# from vega.core.run import init_cluster_args
# from vega.common.general import General

# General._parallel = True
# init_cluster_args()
# master = create_master()

# master.run(trainer)

# # master.run(other trainers)

# master.join()
# shutdown_cluster()
# print("Training is complete. Please check the folder: {}".format(trainer.get_local_worker_path()))
