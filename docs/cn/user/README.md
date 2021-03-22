# 用户指南

## 1. 安装和部署

Vega运行环境为：Ubuntu 18.04、CUDA 10.0、Python3.7，并支持集群部署，详细的安装和部署指导，可参考 **[安装指导](./install.md)** 和 **[集群部署指导](./deployment.md)** 。

## 2. 使用指导

Vega提供了丰富的AutoML任务，用户可根据下表选择合适的算法：

<table>
  <tr><th>任务</th><th>分类</th><th>参考算法</th></tr>
  <tr><td rowspan="3">图像分类</td><td>网络架构搜索</td><td><a href="../algorithms/cars.md">CARS</a>、<a href="../algorithms/nago.md">NAGO</a>、BackboneNas、DartsCNN、GDAS、EfficientNet</td></tr>
  <tr><td>超参优化</td><td><a href="../algorithms/hpo.md">ASHA、BOHB、BOSS、BO、TPE、Random、Random-Pareto</a></td></tr>
  <tr><td>数据增广</td><td><a href="../algorithms/pba.md">PBA</a></td></tr>
  <tr><td rowspan="2">模型压缩</td><td>模型剪枝</td><td><a href="../algorithms/prune_ea.md">Prune-EA</a></td></tr>
  <tr><td>模型量化</td><td><a href="../algorithms/quant_ea.md">Quant-EA</a></td></tr>
  <tr><td rowspan="2">图像超分辨率</td><td>网络架构搜索</td><td><a href="../algorithms/sr_ea.md">SR-EA</a>、<a href="../algorithms/esr_ea.md">ESR-EA</a></td></tr>
  <tr><td>数据增广</td><td><a href="../algorithms/cyclesr.md">CycleSR</a></td></tr>
  <tr><td>图像语义分割</td><td>网络架构搜索</td><td><a href="../algorithms/adelaide_ea.md">Adelaide-EA</a></td></tr>
  <tr><td>物体检测</td><td>网络架构搜索</td><td><a href="../algorithms/sp_nas.md">SP-NAS</a></td></tr>
  <tr><td>车道线检测</td><td>网络架构搜索</td><td><a href="../algorithms/auto_lane.md">Auto-Lane</a></td></tr>
  <tr><td rowspan="2">推荐搜索</td><td>特征选择</td><td><a href="../algorithms/autofis.md">AutoFIS</a></td></tr>
  <tr><td>特征交互建模</td><td><a href="../algorithms/autogroup.md">AutoGroup</a></td></tr>
</table>

Vega针对以上每个算法提供了示例，通过尝试运行示例，可快速掌握各个算法。具体运行示例的方法，可参考 **[示例参考](./examples.md)** 。

通过运行示例，会发现一般Vega算法的输入是数据集和配置信息，输出为网络架构描述、模型训练超参、或模型。同时可根据需要将不同的算法和fully train步骤组合成一个pipeline，实现端到端的AutoML流程，输入数据，即可得到所需的模型。

如上描述Pipeline的详细配置信息，可参考 **[配置指导](./config_reference.md)** 。

## 3. 模型端侧评估

Vega还提供了端侧模型评估的能力，支持的端侧硬件有Davinci推理芯片（ATLAS 200 DK、ATLAS 300产品和开发板环境Evb)和手机，支持在 [Bolt](https://github.com/huawei-noah/bolt) 部署评估。

用户可参考 [评估服务安装和配置指导](./evaluate_service.md) 安装和配置评估服务，用以在架构搜索过程中实时评估搜索到的模型，得到适用于该终端设备的模型。
