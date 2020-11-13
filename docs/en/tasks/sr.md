# Image super-resolution

## 1. Introduction

Image super-resolution is widely used in scenes such as photographing, security, medical imaging, etc., and the visual effect has been greatly improved, but with the widespread use of intelligent hardware, the current image super-distribution network has been in the size and reasoning time of the model Not meeting the requirements, AutoML technology has achieved very good results in how to design a lightweight super-distribution network.

## 2. Algorithm selection

Vega provides two super-resolution network search algorithms, [SR-EA](../ algorithms / sr_ea.md) and [ESR-EA](../ algorithms / esr_ea.md), the two algorithms The goal is the same, it is to search for a light-weight super-point network that is friendly to the end side. The implementation method is different. For the specific difference, please refer to the algorithm documentation of these two algorithms.

It is recommended that users try these two super-score algorithms. For different data sets, the performance of these two algorithms has different results.

## 3. pipeline

For specific pipeline construction, please refer to each algorithm. Also need to pay attention to:

1. Please refer to [Dataset Reference](../developer/datasets.md) to adapt the data set.
2. Configurable evaluation service measures model performance against specific hardware.
