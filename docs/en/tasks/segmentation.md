# Image Segmentation

1. Introduction

The application of image semantic segmentation is more and more extensive, and the requirements for semantic segmentation in the fields of security and autonomous driving are becoming higher and higher. For this reason, the AutoML algorithm also gradually plays an increasingly important role in semantic segmentation.

The backbone network of the semantic segmentation model largely determines the performance of the model. In the field of semantic segmentation, in addition to searching the complete semantic segmentation model directly, you can also search the backbone network of the semantic segmentation network, and then build a complete semantic segmentation network. Way, the latter of which requires more in-depth network background knowledge.

## 2. Algorithm selection

According to the description in the previous chapter, we can divide the application of semantic segmentation into the following scenarios:

1. Scenario A: For ordinary semantic segmentation tasks, users want to provide a semantic segmentation training data set and a test data set to get a suitable semantic segmentation model.
2. Scenario B: For advanced users, search the semantic segmentation network backbone, and then construct the semantic network to get the final semantic segmentation model.

At the same time, the above three may need to deploy models for specific hardware devices, such as the Ascend 310 chip.

### 2.1 Scenario A: Ordinary semantic segmentation task

In this scenario, the Adelaide-EA algorithm can be considered. This algorithm uses an evolutionary algorithm to search a set of semantic segmentation models in a relatively short time. After a fully train, evaluate the performance of each model and select the most The right model.

For specific algorithm introduction, please refer to [Adelaide-EA Algorithm](../algorithms/adelaide_ea.md)

### 2.2 Scenario C: Search semantic segmentation network backbone

Vega provides the SP-NAS algorithm, which is used to search for an efficient backbone network architecture for object detection and semantic segmentation. The use of such algorithms requires a certain amount of deep network architecture knowledge.

For details, please refer to [SP-NAS Algorithm](../algorithms/sp_nas.md)

## 3. pipeline

For specific pipeline construction, please refer to each algorithm. Also need to pay attention to:

1. The data set of semantic segmentation is different, please refer to [Data Set Reference](../developer/datasets.md) to adapt the data set.
2. Configurable evaluation service measures model performance against specific hardware.
