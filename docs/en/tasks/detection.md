# Object detection

## 1. Brief description

At present, more and more intelligent terminals have object detection capabilities, especially in the field of autonomous driving, which puts forward stringent requirements for the network design of object detection. The artificially designed network can no longer keep up with the development of these frontier fields, and AutoML is getting more and more Used in the field of object detection.

The object detection field can be divided into general object detection and object detection for specific application scenarios, such as lane line detection. Such specific object detection methods have certain differences in data sets and algorithms.

The general object detection network can be divided into backbone, neck, rpn, and head. The backbone is mainly responsible for feature extraction and construction. Many object detection algorithms focus on how to construct a more suitable backbone.

## 2. Algorithm selection

According to the description in the previous chapter, we can divide the application of object detection into the following scenarios:

1. Scenario A: For an ordinary object detection task, the user wants to provide an object detection training data set and a test data set to obtain a suitable object detection model.
2. Scenario B: Object detection tasks for specific scenarios, such as lane line detection, etc.
3. Scenario C: For advanced users, search the object detection network backbone, and then construct the object detection network to get the final object detection model.

At the same time, the above three may need to deploy models for specific hardware devices, such as the Ascend 310 chip.

### 2.1 Scenario A: Ordinary object detection task

In this scenario, the SM-NAS algorithm can be considered. The algorithm can be simply understood as two steps. The first step is to determine the appropriate detection network by selecting different combinations of backbone, neck, rpn, and head. Second This step is to conduct a network architecture search for the backbone of the network determined in the first step to obtain a better network architecture.

For specific algorithm introduction, please refer to SM-NAS algorithm (Coming soon)

### 2.2 Scene B: Object detection tasks for specific scenes, such as lane line detection

Vega will provide lane line detection algorithms in the next version, providing specific algorithms for this specific application scenario.

At the same time, Vega will also add more object detection algorithms in specific scenes to provide more choices.

### 2.3 Scenario C: Search object detection network backbone

Vega provides the SP-NAS algorithm, which is used to search for an efficient backbone network architecture for object detection and semantic segmentation. The use of such algorithms requires a certain amount of deep network architecture knowledge.

For details, please refer to [SP-NAS Algorithm](../algorithms/sp_nas.md)

## 3. pipeline

For specific pipeline construction, please refer to each algorithm. Also need to pay attention to:

1. The data sets for object detection are different, please refer to [Data Set Reference](../developer/datasets.md) to adapt the data set.
2. Configurable evaluation service measures model performance against specific hardware.
