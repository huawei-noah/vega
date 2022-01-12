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

### fine tune：convert torchvision weights into spnas backbone.
```yaml

fine_tune:
    pipe_step:
        type: TrainPipeStep

    model:
        pretrained_model_file: /cache/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth  # torchvision weights file
        model_desc:
            type: FasterRCNN
            convert_pretrained: True     # convert weights into SerialBackbone
            backbone:
                type: SerialBackbone     # backbone type

```

### step1: Serial-level

```yaml
    search_algorithm:
        type: SpNasS
        max_sample: 20              # Maximum number of adopted structures
        objective_keys: ['mAP', 'params']   # Objective keys for pareto front
        num_mutate: 3               # Maximum number of mutate blocks
        add_stage_ratio: 0.05       # Probability of the number of new feature layers
        expend_ratio: 0.3           # Probability of the number of new blocks
        max_stages: 6               # Maximum number of allowed feature layers
    
    search_space:
        type: SearchSpace
        hyperparameters:
            -   key: network.backbone.code   # Search space
                type: CATEGORY
                range: ['111-2111-211111-211']

    model:
        pretrained_model_file: "{local_base_path}/output/fine_tune/model_0.pth"   # Get weight file from fine_tune pipe step
        model_desc:
            type: FasterRCNN         
            freeze_swap_keys: True   # Freeze not swap layers 
            backbone:                # block type
                type: SerialBackbone
    
```

### step2: Reignition

```yaml
    pipe_step:
        type: TrainPipeStep
        models_folder: "{local_base_path}/output/serial/"  # Get desc file from serial pipe step

    trainer:
        type: Trainer
        callbacks: ReignitionCallback   # Do reignition
```

### Step3: Parallel-level

```yaml
     pipe_step:
        type: SearchPipeStep
        models_folder: "{local_base_path}/output/reignition/"  # Get desc file from reignition pipe step

    search_algorithm:
        type: SpNasP
        max_sample: 10

    model:
        pretrained_model_file:  "{local_base_path}/output/fine_tune/model_0.pth"  # Load fasterrcnn weights file
        model_desc:
            type: FasterRCNN
            neck:
              type: ParallelFPN  # Neck type

    search_space:
        type: SearchSpace
        hyperparameters:
            -   key: network.neck.code   # Search Space of neck
                type: CATEGORY
                range: [[0, 1, 2, 3]]
```

### step4：fully train

```yaml
    pipe_step:
        type: TrainPipeStep
        models_folder: "{local_base_path}/output/parallel/"  # Get desc file and weights file from parallel pipe step
```

### Algorithm output

- The optimal models with fully training.
- Logs of all models during the entire search process, and logs for models from the Pareto front({local_base_path}/output).

## Benchmark

Benchmark configuration: [spnas.yml](https://github.com/huawei-noah/vega/blob/master/examples/nas/sp_nas/spnas.yml)
