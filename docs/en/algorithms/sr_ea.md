# SR-EA

## Algorithm Introduction

SR-EA is a module that uses the evolutionary algorithm (EA) to search for the image super-resolution (SR) network architecture. EA is a common automatic network architecture search method (NAS). The search process is as follows：

1. Sampling a series of models (usually random), and perform incomplete training (for example, reduce the number of iterations or training samples) on each model.
2. Calculating a Pareto front (pareto front) of all currently generated models, generating an evolutionary model based on the Pareto front, and performing incomplete training on each model;
3. Repeat step 2 until the specified maximum iterations are reached or the specified performance is achieved.

## Algorithm Principle

SR-EA provides two series of network architectures: modified SRResNet (baseline) and CCRN-NAS (from Noah's Ark ). The following figure shows the structure of Modified SRResNet:

![Modified SRResNet](../../images/sr_ea_SRResNet.png)

SR-EA provides two architecture search policies ( random search & brute force search ), which focus on searching for the number of blocks and channels in the architecture.
CCRN-NAS is a network architecture dedicated to lightweight networks. The CCRN-NAS consists of three types of blocks:

1. Residual block whose kernel size is 2;
2. Residual block whose kernel size is 3;
3. Channel Acquisition Block (CIB): consists of the two modules in sequence. Each module combines the above two outputs in the channel dimension. Therefore, the number of channels is doubled after the CIB.

Pipeline provides a sample for CCRN-NAS architecture search. It searches for the combination of the three modules to optimize the network architecture.

## Search Space and Search Policy

The search space of the modified SRResNet includes the number of blocks and channels. We provide two search methods: random search (RS) and brute force (BF). In the two search methods, users need to define the range of the block number and the channel number for each convolution layer. RS generates model randomly from these range until the number of models reaches max_count. On the other size, BF will train all selected models.

The search space of CCRN-NAS is a combination of three types of blocks:

1. Random search: The number of residual blocks and the number of CIBs are selected based on user-defined conditions. In the residual block, the ratio of convolution layer with kernel size 2 is randomly generated between [0,1]. The sampling process generates a common residual block and randomly inserts CIB into the residual block.

2. Evolution search: Models on Pareto front are selected for modification each time. Following operations are allowed:
  – Change the kernel size of a random residual block from 2 to 3 or from 3 to 2.
  – A residual block is added to the random number of layers, and the kernel size is randomly generated in 2 and 3.

## Configuring the search space

The configuration of search space and search algorithm is as follows:

```yaml
pipeline: [random, mutate]

random:
    pipe_step:
        type: SearchPipeStep

    search_space:
        type: SearchSpace
        modules: ['custom']
        custom:
            type: MtMSR
            in_channel: 3
            out_channel: 3
            upscale: 2
            rgb_mean: [0.4040, 0.4371, 0.4488]
            candidates: [res2, res3]
            block_range: [10, 80]
            cib_range: [3, 4]

    search_algorithm:
        type: SRRandom
        codec: SRCodec
        policy:
            mum_sample: 1000

mutate:
    search_space:
        ref: random.search_space

    search_algorithm:
        type: SRMutate
        codec: SRCodec
        policy:
            mum_sample: 1000
            num_mutate: 3
```

### Output

The outputs are as follows:

• The model on the found Pareto front after fully training.
• Logs of all models in random search and evolutionary search process (result.csv)
• Logs of Pareto front results (pareto_front.csv).
