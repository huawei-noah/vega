# SEGMENTATION-Adelaide-EA-NAS

## 1. Algorithm Introduction

SEGMENTATION-Adelaide-EA-NAS is an network architecture search algorithm for semantic segmentation. The search algorithm is improved based on the Fast-NAS developed by Shen's team in Adelaide, and the search policy is changed from reinforcement learning (RL) to evolutionary search (EA). Therefore, the search algorithm is named SEGMENTATION-Adelaide-EA-NAS. For more details about Fast-NAS, see the article on arxiv at <https://arxiv.org/abs/1810.10804> .

## 2. Algorithm Principles

A semantic segmentation model may be generally decoupled into an encoder part (which may also be referred to as a backbone) and a decoder part. Generally, the backbone generates feature maps with different scales, and the decoder implements pixel-level classification, by selecting feature maps of different scales for fusion and finally upsampling to the original input resolution. The SEGMENTATION-Adelaide-EA-NAS is used to search for decoders.

![F3](../../images/Adelaide-EA-NAS1.jpg)

### 2.1 Search Space and Strategy

The decoder for segmentation model is searched, which includes (a) Connection between the decoder and the backbone; (b)Structure of the cell (including the operators and connection inside the cell); (3)Connections between cells. The EA search policy is used.

Each network is represented by 19 characters. The first 13 characters indicate the structure of the cell in the decoder (including the operator and connection mode in the cell), and the last 6 characters indicate the connection between the decoder and the backbone and the connection of the cell in the decoder. In each generation of search, a character in the previous generation is changed to obtain a new network. In addition, we need to pre-train the backbone part before the search.  The purpose of this is to speed up the convergence of the model, thereby speeding up the search.

The SEGMENTATION-Adelaide-EA-NAS search space is described as follows:

#### 2.1.1 Searching for a cell

The search space of SEGMENTATION-Adelaide-EA-NAS includes the following operators::

`conv1x1, conv3x3, sep_conv_3x3, sep_conv_5x5, conv3x3_dil3, conv3x3_dil12, sep_conv_3x3_dil3, sep_conv_5x5_dil6`

As shown in the following figure, two cells are fused in two ways: concatenation or addition. The number of channels of each cell is fixed. When the two feature maps with different resolutions are fused, the feature map of the low resolution need to be upsampled to the size of the feature map of the high resolution. The fusion mode and the number of channels of a cell can be obtained from agg_size and agg_concat in the configuration.

![F3](../../images/Adelaide-EA-NAS3.jpg)

Each model consists of 19 characters. The first 13 characters indicate the cell structure (including the operator and connection mode) of the decoder, which can be parsed as follows:

`[op1, [1, 0, op2, op3], [4, 3, op4, op5], [2, 0, op6, op7]]`

op1 to op7 indicate operators. For example, if op1 is 3, the third operator in op_names (the operator ID starts from 0) is used. The definition of op_names in the configuration is as follows:

`[conv1x1, conv3x3, sep_conv_3x3, sep_conv_5x5, conv3x3_dil3, conv3x3_dil12, sep_conv_3x3_dil3, sep_conv_5x5_dil6]`

, 3 indicates the sep_conv_5x5 operator. Besides op1 to op7, there are six digits, indicating the output from which the input of the corresponding operator comes. For example, in the above example, op2 corresponds to No.1 output, op3 corresponds to No.0 output, op4 corresponds to No.4 output, op5 corresponds to No.3 output, and op6 corresponds to No.2 output, op7 corresponds to output 0.  Output from 0 to 9 is the operator or symbol marked in orange in the following figure.

![F3](../../images/Adelaide-EA-NAS4.jpg)


#### 2.1.2 Searching for a connection

The search for connection includes the connection between the decoder and the backbone and the connection between the decoder cells. The connection can be encoded by six characters, which can be  parsed as follows:

`[[cell0,cell1], [cell2,cell3], [cell4,cell5]]`
The following figure shows the form of [[2, 3], [3, 1], [4, 4]]. [2,3] indicates that the No. 2 feature map and No. 3 feature map are fused. The fusion contains the following: a ) No. 2/3 feature map are fed into a cell; b)the method of MergeCell is invoked to fuse the two feature maps to obtain the No. 4 feature map. Correspondingly, [3, 1] and [4, 4] are fused in the same way. Finally, concat and conv 1x1 are performed on all fused feature maps to obtain the network output.

![F3](../../images/Adelaide-EA-NAS5.jpg)

Particularly, all feature maps in the backbone are convolved by using a conv 1x1 to make the output channel to be agg_size in the configuration.

### 2.2 Configuring the search space

Parameters are set in the configuration file (examples/nas/adelaide_ea/adelaide_ea.yml).

```yaml

pipeline: [random, mutate]

random:
    search_space:
        type: SearchSpace
        modules: ['custom']
        custom:
            type: AdelaideFastNAS
            backbone_load_path: /cache/models/mobilenet_v2-b0353104.pth
            backbone_out_sizes: [24, 32, 96, 320]
            op_names: [conv1x1, conv3x3, sep_conv_3x3, sep_conv_5x5, conv3x3_dil3, sep_conv_3x3_dil3, sep_conv_5x5_dil6]
            agg_size: 64
            sep_repeats: 1      # Number of times for repeated convolutional separation in the decoder cell
            agg_concat: true    # If the value is TRUE, the feature fusion mode in the decoder is concat. Otherwise, the feature fusion mode is +
            num_classes: 21
    search_algorithm:
        type: AdelaideRandom
        codec: AdelaideCodec
        max_sample: 100

mutate:
    search_space:
        ref: random.search_space
    search_algorithm:
        type: AdelaideMutate
        codec: AdelaideCodec
        max_sample: 100
        pareto_front_file: "{local_base_path}/output/random/pareto_front.csv"
        random_file: "{local_base_path}/output/random/random.csv"
```

## 3. Dataset

The dataset for image semantic segmentation needs to include RGB images and corresponding mask tags. The RGB image value ranges from 0 to 255, and the mask label ranges from 0 to N-1 (N indicates the number of segmentation types. 255 indicates the ignored label, which can be set in the configuration file). The corresponding data list is a .txt file. Each line indicates the RGB image and the corresponding label file, a space character is used for separation. For example:

`VOC2012/JPEGImages/2007_000033.jpg VOC2012/SegmentationClassAug/2007_000033.png`

### 4. Output

The output includes a series of .pth files (models trained to the num_iter iteration times in the configuration file), the result.csv file, and the pause_front.csv file. The result.csv file records all search models, and the pareto_front.csv file records all pareto_front models. The .csv file contains encoding, flops, parameters, and mIOU.

1. encoding: A 19-character string indicates the structure of a model, which ends with an underscore (_) to avoid record errors caused by encoding starting with 0.
2. flops: Records the macc value of the model. For example, 1371603728 indicates 1.277 GB.
3. parameters: Records the values of parameters in the model. For example, 3162900 indicates 3.016 MB.
4. mIOU: Records the segmentation performance.

## 5. Benchmark

For details, see the benchmark configuration item in the [adelaide_ea.yml](https://github.com/huawei-noah/vega/blob/master/examples/nas/adelaide_ea/adelaide_ea.yml) configuration file.
