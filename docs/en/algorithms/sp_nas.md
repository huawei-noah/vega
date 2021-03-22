# SP-NAS (Serial-to-Parallel Backbone Search for Object Detection)

## Algorithm Introduction

SP-NAS is an efficient architecture search algorithm for object detection and semantic segmentation based on the backbone network architecture. The existing object detectors usually use the feature extraction network designed and pre-trained on the image classification task as the backbone. We propose an efficient, flexible and task-oriented search scheme based on NAS. which is a two-phase search solution from serial to parallel to reduce repeated ImageNet pre-training or long-time training from scratch.

## Algorithm Principles

This method has two phases:

1. In serial phase, the block sequence with optimal scaling ratio and output channel is found by using the "swap-expand-reignite" search policy. This search policy can guranteen a new searched architecture to completely inherit of weight from arichtectures before morphism.
2. In parallel phase, parallized network structures are designed, sub-networks integrated by different feature layers are searched to better fuse the high-level and low-level semantic features. The following figure shows the search policy.

![sp-nas](../../images/sp_nas.png)

## Search Space and Search Policy

**Serial-level**

- Swap-expand-reignite policy:  Growing starts from a small network to avoid repeated ImageNet pre-training.
  - The new candidate network is obtained by "switching" or "expanding" the grown network for many times.
  - Quickly train and evaluate candidate networks based on inherited parameters.
  - When the growth reaches the bottleneck, the network is re-trained using ImageNet. The number of ignition times is no more than 2.

- Constrained optimal network: A serial network with limited network resources (latency, video memory usage, or complexity) is selected to obtain the maximum performance.

- Search space configuration:
  - Block type: Basic Block, BottleNeck Block, and ResNext;
  - Network depth: 8 to 60 blocks;
  - Number of stages: 5 to 7;
  - Width: Position where the channel size is doubled in the entire sequence.

**Parallel-level**

- Based on the result SerialNet from the serial search phase (or the existing handcraft serial network such as ResNet series), search for the parallel structure stacked on SerialNet to better utilize and fuse feature information with different resolutions from different feature layers.
- Search policy: Random sampling to meet the resource constraints: The probability of adding additional subnets is inversely proportional to the FLOPS of the subnets to be added.
- Search space: SerialNet is divided into L self-networks based on the number of feature layers and K sub-networks are searched for in each phase.

## Usage Guide

### Example 1: Serial phase

```yaml
search_algorithm:
    type: SpNas
    codec: SpNasCodec
    total_list: 'total_list_s.csv'  # Record the search result.
    sample_level: 'serial'          # Serial search: 'serial', parallel search: 'parallel'
    max_sample: 10      # Maximum number of adopted structures
    max_optimal: 5      # The top 5 seed networks are reserved in the serial phase and start to mutate, set the number of parallel phases to 1
    serial_settings:
         num_mutate: 3
         addstage_ratio: 0.05   # Probability of the number of new feature layers
         expend_ratio: 0.3      # Probability of the number of new blocks
         max_stages: 6          # Maximum number of allowed feature layers
    regnition: False            # Whether ImageNet has been performed. regnite#
#    last_search_result: # Whether to search for the config epoch of the
search_space:
    type: SearchSpace
    config_template_file: ./faster_rcnn_r50_fpn_1x.py   # starting point network based on the existing search records
    epoch: 1        # Number of fast trainings for each sampling structure
```

### Example 2: Parallel phase

```yaml
search_algorithm:
    type: SpNas
    codec: SpNasCodec
    total_list: 'total_list_p.csv'  # Record the search result.
    sample_level: 'parallel'        # Serial search:'serial', parallel search: 'parallel'
    max_sample: 10      # Maximum number of structures
    max_optimal: 1
    serial_settings:
         last_search_result: 'total_list_s.csv'     # Search based on existing search records.
         regnition: False   # Whether the ImageNet regnite
search_space:
    type: SearchSpace
    config_template_file: ./faster_rcnn_r50_fpn_1x.py   # start point network is configured. config
    epoch: 1        # Each sampling Fast training data of structure
```

### Example 3: Fully train

**Completely train the best network based on the search records.**

```yaml
trainer:
    type: SpNasTrainer
    gpus: 8
    model_desc_file: 'total_list_p.csv' 
    config_template: "./faster_rcnn_r50_fpn_1x.py"
    regnition: False    # Whether ImageNet regnite
    epoch: 12
    debug: False
```

**Fully trained optimal network based on network coding**

```yaml
trainer:
    type: SpNasTrainer
    gpus: 8
    model_desc_file: "{local_base_path}/output/total_list_p.csv"
    config_template: "./faster_rcnn_r50_fpn_1x.py"
    regnition: False    # Whether ImageNet regnite
    epoch: 12
    debug: False
```

### Algorithm output

- The optimal models with fully training.
- Logs of all models during the entire search process, and logs for models from the Pareto front(pareto_front.csv).

## Benchmark

Benchmark configuration: [sp_nas.yml](https://github.com/huawei-noah/vega/tree/master/examples/nas/sp_nas.yml)
