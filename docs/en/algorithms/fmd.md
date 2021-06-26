# A Guidance of Feature Map Distortion (FMD)

## 1. Introduction

Deep neural network (DNN) usually has a large quantity of redundant parameters, which makes the performance superior to the traindional human-designed features, but also incurs the overfitting problem, reduces generalization ofthe network. Various methods are proposed to alleviate the overfitting problem, which can reduce the perfomance gap between the training/test set without obviously performance drop on the training set, thereby improving the performance of the neural network.

There are a lot of researches on network regularization. The most widely used method is the random inactivation (Dropout). In recent years, many variant methods based on random inactivation have been proposed and achieve good results. However, this type of inactivation operation randomly discards some neurons in the neural network, and sets the output of the corresponding neuron to 0. This operation is a manually defined operation, which greatly destroys the representation capability of the network.

In addition to the direct set-to-zero operation, more flexible perturbation can be performed on the output of the neuron, which can effectively reduce the complexity of the model while minimizing the damage to the network representation, thereby improving the generalization of the network. We focus on how to perturb the output of neurons to alleviate the overfitting phenomenon and obtain a deep neural network with better generalization capability.

The proposed method considers a flexible and universal neuron output disturbance. During the training , the network parameters and perturbation terms are optimized alternately. The network parameters are optimized by minimizing a common loss function (for example, a cross-entropy loss function in classification). Given the parameters of the current network, the disturbance term is obtained by minimizing empirical rademacher complexity optimization, and the disturbance term is used to interfere with the forward process in the next iteration. The experimental results show that the network trained by the neuron output perturbation method has better performance in the test set.

![F3](../../images/fmd_framework.PNG)

This method can be considered as an operator that replaces dropout. Generally, this operator is used after the convolution layer (you can add this operator after the bn or relu layer).

## 2. Operator Description

The code related to the FMD operator is stored in the following directory:

```text
vega/networks/pytorch/ops/fmdunit.py
```

The parameters related to the FMD operator are as follows:

| Parameter | Description |
| :-- | :-- |
| drop_prob  | probability of perturbating the neuron. |
| alpha | Adjust the intensity of disturbances. |
| block_size | size of neuron blocks that may be perturbed each time. |

**drop scheduler**The drop scheduler is used to adjust the drop rate. Specifically, starting from start_value, the preset drop value is reached through nr_steps. It is defined

```yaml
dropblock: instantiation of the FMD operator.
start_value: initial drop rate.
stop_value: final drop rate.
nr_steps: number of steps required from the initial value to the final value.
```

## 3. Example

The code example for using the FMD operator to construct a neural network is as follows:

```text
example/fully_train/fmd/networks/resnet_cifar.py
```

Configure the configuration file correctly and call the main function to run the pipeline. For details, see [Example Reference](../user/examples.md).

This method uses the default parameters in the .yml file in the pipeline to achieve a precision of 94.50% in the cifar10 data set. The accuracy values shown in the original paper are as follows:

![F3](../../images/fmd_2.PNG)

## 4. Network customization

You may need to customize the network structure. The .py file of the customized network structure should be placed in the networks folder, and the corresponding network name should be modified in the .yml configuration file.

For details about the network definition, see the resnet_cifar.py file. For details about how to call the fmd operator, see line 135 for reference.

![F3](../../images/fmd_3.PNG)

In addition, you need to modify the init and forward operations of the network. Specifically, for the init method, reference is in lines 196 to 210, which are the the fmd layer parameters and the conv layer parameters.

![F3](../../images/fmd_4.PNG)

In the foward method, refer to lines 249-255, which indicates that the weight_behind parameter of the FMD layer is assigned a value and the drop rate is adjusted so that the drop rate changes from 0 to the specified value.

![F3](../../images/fmd_5.PNG)
