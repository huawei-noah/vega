# Efficient Residual Dense Block Search for Image Super-resolution (ESR_EA)

## 1. Algorithm Introduction

 Taking the advantage of the rapid development of GPU and deep convolutional network (DCNN), the visual quality of super-resolution is greatly improved, which makes image super-resolution widely used in real life.

At the same time, the model size of the super resolution network is increasing from 57K to 43M, and the computing workload reaches 10192G FLOPs (RDN). Meanwhile, the computing and storage budgets of mobile devices are limited, which constrains the application of the huge super-resolution models on mobile devices (for example, mobile phones, cameras, and smart homes). A lightweight super-resolution model is appealed for mobile applications.

Common methods for compressing a super-resolution model can be classified into two categories: a) manually designed efficient structural units (such as group convolution and recuresive); b) automatically search a lightweight entwork architecture. The existing architecture search algorithms are mainly focus on using convolution units and connections to search lightweight networks. However, the obtained network structure is very irregular and is not hardware friendly. Moreover, the entire backbone is calculated on a single scale, which means a huge computation workload.

We propose a network architecture search algorithm , which constructs a modular search space, takes the parameters and computations as constraints, and the network accuracy (PSNR) as the objective to search for a lightweight and fast super-resolution model. In this way, the network structure is hardware friendly. In addition, we compress the super-resolution network from three aspects: channel, convolution, and feature scale. The proposed algorithm has been published at AAAI 2020.

```text
[1] Song, D.; Xu, C.; Jia, X.; Chen, Y.; Xu, C.; Wang, Y. Efficient Residual Dense Block Search for Image Super-Resolution[C]. AAAI 2020.
```

## 2. Algorithm Description

Firstly, the algorithm constructs a search space based on modules, takes the parameters and computations as constraints, and the network accuracy (PSNR) as the objective to search for an efficient super-resolution network structure. In addition, a high efficiency super-resolution module based on RDB is designed to compress the redundant information of super network from channel, convolution and characteristic scale. Finally, genetic algorithm is used to search for the number of each type of module, the corresponding location and the specific internal parameters. The following figure shows the algorithm framework.

![arch](../../images/esr_arch.png)

We take RDN as the basic network structure and Efficient Dense Block (RDB) as the basic module, and searches for the number, the types and the internal parameters of the modules. You can assign the compression ratio of each module and the location of each module in the whole network during the search. We design three kinds of efficient residual-intensive modules, which compress the redundancy of channel, convolution and feature scale respectively. The detailed network structure is as follows:

![block](../../images/esr_block.png)

The proposed algorithm has two steps : network structure search and full training. In order to speed up the search, the model evaluation is usually acheived by means of fast training. Fully train on a large data set needs to be performed after we have the searched candidates.

The following is an example of the searched:

```text
['G_8_16_24', 'C_8_16_16', 'G_4_16_24', 'G_8_16_16', 'S_4_24_32', 'C_8_16_48', 'S_4_16_24', 'G_6_16_24', 'G_8_16_16', 'C_8_16_24', 'S_8_16_16', 'S_8_16_24', 'S_8_16_32', 'S_6_16_16', 'G_6_16_64', 'G_8_16_16', 'S_8_16_32']
```

### 2.1 Search Space and Searching Strategy

The efficient RDB is used as the basic modules for search. Considering the hardware efficiency, the number of convolutional channels is also a multiple of 16, for example, 16, 24, 32, 48, or 64. The algorithm mainly searches for the number of modules, the type of each location module, and the specific parameters (such as the number of convolutions and channels) in the model, and allocates the compression ratio of each aspect.

The search strategy is mainly based on evolutionary algorithms. Firstly, RDN is used as the basic structure framework to encode the global network, and a next generation population is generated through crossover and mutation. There are two selection modes for parents selection , which are tourament and roulette modes. The final high performance network structure is obtained through iterative evolution.

### 2.2 Configuring

For details, see the configuration file examples/nas/esr_ea/esr_ea.yml in the sample code.

```yaml
nas:
    search_space:                       # Set the network structure search parameters.
        type: SearchSpace
        modules: ['esrbody']
        esrbody:
            type: ESRN
            block_type: [S,G,C]         # module
            conv_num: [4,6,8]           # Number of convolutions in the module
            growth_rate: [8,16,24,32]   # Number of convolutional channels in the module
            type_prob: [1,1,1]          # Probability of module type selection
            conv_prob: [1,1,1]          # Probability of selecting the number of convolutions
            growth_prob: [1,1,1,1]      # Probability of selecting the number of convolution channel
            G0: 32                      # Number of initial convolution channels
            scale: 2                    # Scale of the super-distribution
    search_algorithm:
        type: ESRSearch
        codec: ESRCodec
        policy:
            num_generation: 20          # Number of iterations in evolution algorithm
            num_individual: 8           # Number of individuals in evolution algorithm
            num_elitism: 4              # Number of elites to be reserved
            mutation_rate: 0.05         # probability of mutation for each gene
        range:
            node_num: 20                # Upper limit of the modules
            min_active: 16              # Lower limit of the modules
            max_params: 325000          # Maximum parameters of network
            min_params: 315000          # Minimum parameters of network


```

If other blocks need to be used as the basic modular structure or multiple types of blocks need to be searched.

## 3. Application Scenarios

This method is used to search for low-level vision tasks, such as super-resolution, denoising, and de-mosaic. Currently, the RGB data is used for training. We use the DIV2K dataset in this repo. Other similar datasets can be used by simply change the config.

## 4. User Guide

You can refer to the example code examples/nas/esr_ea. In this folder, the esr_ea.yml configures all setting required for running the ESR code. In addition to search_space and search_algorithm described in the above section, the following configurations are also need to be set:

If you want to speed up the search or debug the code before running the search and training for a long time, you can reduce the parameter values in the following setting:

```yaml
nas:
    search_algorithm:               # search algorithm parameter setting
        policy:
            num_generation: 20      # Number of iterations of the evolution algorithm
            num_individual: 8       # Number of individuals in each population
    trainer:                        # Parameter setting for model training during search
        epochs: 500                 # Number of training iterations
```

To improve the model precision, increase the values of the following parameters:

```yaml
nas:
    search_algorithm:               # search algorithm parameter setting
        policy:
            num_generation: 20      # Number of iterations of the evolution algorithm
            num_individual: 8       # Number of individuals in each population
    trainer:                        # Parameter setting for model training during the search process
        epochs: 500                 # Number of training iterations

fully_train:
    trainer:                        # Parameter settings for model training during the entire process
        epochs: 15000               # Training iteration times
        lr_scheduler:
            type: MultiStepLR
            milestones: [8000,12000,13500,14500]    # Learning rate attenuation position
            gamma: 0.5

```

Generally, a better performance can be obtained by using a set of candidates structure for training instead of a single structure.

After the parameters are adjusted, run the pipeline by referring to the [example reference](../user/examples.md).

## 5. Algorithm output

The following results can be obtained based on the default configuration:

![result](../../images/esr_results.png)

```text
[1] Chu, X.; Zhang, B.; Ma, H.; Xu, R.; Li, J.; and Li, Q. 2019. Fast, accurate and lightweight super-resolution with neural architecture search. arXiv preprint arXiv:1901.07261.
```
