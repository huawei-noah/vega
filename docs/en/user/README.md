# User Guide

## 1. Installation and Deployment

The Vega running environment is Ubuntu 18.04, CUDA 10.0, and Python 3.7, and supports cluster deployment. For details about the installation and deployment, see **[Installation Guide](./install.md)** and **[Cluster Deployment Guide](./deployment.md)**.

## 2. Usage Guide

Vega provides various AutoML tasks. You can select an appropriate algorithm based on the following table.

<table>
  <tr><th>Task</th><th>categorize</th><th>Algorithms</th></tr>
  <tr><td rowspan="3">Image Classification</td><td>Network Architecture Search</td><td><a href="../algorithms/cars.md">CARS</a>, <a href="../algorithms/nago.md">NAGO</a>, BackboneNas, DartsCNN, GDAS, EfficientNet</td></tr>
  <tr><td> Hyperparameter Optimization</td><td><a href="../algorithms/hpo.md">ASHA, BOHB, BOSS, BO, TPE, Random, Random-Pareto</a></td></tr>
  <tr><td>Data Augmentation</td><td><a href="../algorithms/pba.md">PBA</a></td></tr>
  <tr><td rowspan="2">Model Compression</td><td>Model Pruning</td><td><a href="../algorithms/prune_ea.md">Prune-EA</a></td></tr>
  <tr><td>Model Quantization</td><td><a href="../algorithms/quant_ea.md">Quant-EA</a></td></tr>
  <tr><td rowspan="2">Image Super-Resolution</td><td>Network Architecture Search</td><td><a href="../algorithms/sr_ea.md">SR-EA</a>, <a href="../algorithms/esr_ea.md">ESR-EA</a></td></tr>
  <tr><td>Data Augmentation</td><td><a href="../algorithms/cyclesr.md">CycleSR</a></td></tr>
  <tr><td>Image Segmentation</td><td>Network Architecture Search</td><td><a href="../algorithms/adelaide_ea.md">Adelaide-EA</a></td></tr>
  <tr><td>Object Detection</td><td>Network Architecture Search</td><td><a href="../algorithms/sp_nas.md">SP-NAS</a></td></tr>
  <tr><td>Lane Detection</td><td>Network Architecture Search</td><td><a href="../algorithms/auto_lane.md">Auto-Lane</a></td></tr>
  <tr><td rowspan="2">Recommender System</td><td>Feature Selection</td><td><a href="../algorithms/autofis.md">AutoFIS</a></td></tr>
  <tr><td>Feature Interactions Selection</td><td><a href="../algorithms/autogroup.md">AutoGroup</a></td></tr>

Vega provides examples for each of the above algorithms. You can quickly master each algorithm by running the examples. For details about how to run the example, see **[Example Reference](./examples.md)**.

The following example shows that the input of the Vega algorithm is the data set and configuration information, and the output is the network architecture description, model training hyperparameter, or model. In addition, different algorithms and fully train steps can be combined into a pipeline to implement the E2E AutoML process. You can enter data to obtain the required model.

For details about the pipeline configuration, see **[Configuration Guide](./config_reference.md)**.

## 3. Model Evaluation On Device

Vega also provides the model evaluation capability. The device hardware supported by Vega includes Davinci inference chips (Atlas 200 DK, Atlas 300, and development board environment Evb) and mobile phones. The [Bolt](https://github.com/huawei-noah/bolt) can be deployed for evaluation.

You can install and configure the evaluation service by referring to [Evaluation Service Installation and Configuration Guide](./evaluate_service.md) to evaluate the model found during architecture search in real time and obtain the model applicable to the device.
