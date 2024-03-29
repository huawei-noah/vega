# benchmark

## environment

Hardware environment:

* GPU: V100 * 8
* CPU: Intel Xeon

Software environment:

* Ptyhon 3.7
* PyTorch 1.3
* CUDA 10.0

## Image Classification on ImageNet

| Method           | Model Name       | Accuracy   | Paras(M) |
| ------------     | ------------     | --------   | -----    |
| **EfficientNet** | **B0**           | **76.83**  |          |
|                  | **B4**           | **82.8**   |          |
|                  | **B8 672**       | **85.7**   |  88      |
|                  | **B8 832**       | **85.8**   |  88      |
| DARTS            | -                | 73.30      | -        |
| AmeobaNet-A      | -                | 83.90      | -        |
| ProxylessNAS     | -                | 75.10      | -        |
| StacNAS          | -                | 76.78      | -        |

## Image Classification on Cifar-10

| Method       | Model Name | #Paras(M) | Accuracy |
| ------------ | ---------- | --------- | -------- |
| **CARS**     | **CARS-A** | **1.402** | **95.92**|
|              | **CARS-B** | **1.697** | **96.58**|
|              | **CARS-C** | **1.913** | **96.74**|
|              | **CARS-D** | **2.225** | **97.05**|
|              | **CARS-E** | **2.408** | **97.25**|
|              | **CARS-F** | **3.767** | **97.30**|
|              | **CARS-G** | **4.377** | **97.38**|
|              | **CARS-H** | **4.506** | **97.43**|
| DARTS        | -          | 3.30      | 97.24    |
| NSGANet      | -          | 3.30      | 97.25    |
| SNAS         | Aggressive | 2.30      | 96.90    |
|              | Mild       | 2.90      | 97.02    |
| AmeobaNet-A  | -          | 3.10      | 96.88    |
| ProxylessNAS | -          | 5.70      | 97.92    |
| StacNAS      | -          | 3.90      | 97.98    |

## Detection on CULane

| Method       | Model Name | FLOPs(G) | Params| F1 Score |
| ------------ | ---------- | --------- | -------- |-------- |
| **AutoLane**  | **CULane-S** | **2.09** | **4.57** | **71.5** |
|               | **CULane-M** | **8.54** | **6.6** | **74.6** |
|               | **CULane-L** | **2.08** | **7.32** | **75.2** |
| SCNN      | - | 328.4 | - | 71.6 |
| SAD       | - | 162.2 | - | 71.8 |
| PointLane | - | 25.1  | - | 70.2 |

## Super-Resolution on Set5

| Method     | Model Name   | Model Size/M | Flops/G    | PSNR      | SSIM       |
| -------    | ----------   | ------------ | -------    | -----     | ------     |
| **ESR-EA** | **ESRN-V-1** | **1.32**     | **40.616** | **37.79** | **0.9566** |
|            | **ESRN-V-2** | **1.31**     | **40.21**  | **37.84** | **0.9569** |
|            | **ESRN-V-3** | **1.31**     | **41.676** | **37.79** | **0.9570** |
|            | **ESRN-V-4** | **1.35**     | **40.17**  | **37.83** | **0.9567** |
| **SR_EA**  | **M2Mx2-A**  | **3.20**     | **196.27** | **38.06** | **0.9588** |
|            | **M2Mx2-B**  | **0.61**     | **35.03**  | **37.73** | **0.9562** |
|            | **M2Mx2-C**  | **0.24**     | **13.49**  | **37.56** | **0.9556** |
| SRCNN      | -            | -            | 52.7       | 36.66     | 0.9524     |
| CARN-M     | -            | -            | 91.2       | 37.53     | 0.9583     |
| FALSR-B    | -            | 0.32         | 74.70      | 37.61     | 0.9585     |

## Super-Resolution on Set14

| Method     | Model Name   | Model Size/M | Flops/G    | PSNR      | SSIM       |
| -------    | ----------   | ------------ | -------    | -----     | ------     |
| **ESR-EA** | **ESRN-V-1** | **1.32**     | **40.616** | **33.37** | **0.8887** |
|            | **ESRN-V-2** | **1.31**     | **40.21**  | **33.37** | **0.8911** |
|            | **ESRN-V-3** | **1.31**     | **41.676** | **33.35** | **0.8878** |
|            | **ESRN-V-4** | **1.35**     | **40.17**  | **33.35** | **0.8902** |
| **SR_EA**  | **M2Mx2-A**  | **3.20**     | **196.27** | **33.65** | **0.8943** |
|            | **M2Mx2-B**  | **0.61**     | **35.03**  | **33.32** | **0.8870** |
|            | **M2Mx2-C**  | **0.24**     | **13.49**  | **33.13** | **0.8829** |
| SRCNN      | -            | -            | 52.7       | 32.42     | 0.9063     |
| CARN-M     | -            | -            | 91.2       | 33.26     | 0.9141     |
| FALSR-B    | -            | 0.32         | 74.70      | 33.29     | 0.9143     |

## Super-Resolution on B100

| Method     | Model Name   | Model Size/M | Flops/G    | PSNR      | SSIM       |
| -------    | ----------   | ------------ | -------    | -----     | ------     |
| **ESR-EA** | **ESRN-V-1** | **1.32**     | **40.616** | **32.09** | **0.8802** |
|            | **ESRN-V-2** | **1.31**     | **40.21**  | **32.08** | **0.8810** |
|            | **ESRN-V-3** | **1.31**     | **41.676** | **32.05** | **0.8789** |
|            | **ESRN-V-4** | **1.35**     | **40.17**  | **32.06** | **0.8810** |
| **SR_EA**  | **M2Mx2-A**  | **3.20**     | **196.27** | **32.20** | **0.8842** |
|            | **M2Mx2-B**  | **0.61**     | **35.03**  | **32.00** | **0.8989** |
|            | **M2Mx2-C**  | **0.24**     | **13.49**  | **31.89** | **0.8783** |
| SRCNN      | -            | -            | 52.7       | 31.26     | 0.8879     |
| CARN-M     | -            | -            | 91.2       | 31.92     | 0.8960     |
| FALSR-B    | -            | 0.32         | 74.70      | 31.97     | 0.8967     |

## Super-Resolution on Urban100

| Method     | Model Name   | Model Size/M | Flops/G    | PSNR      | SSIM       |
| -------    | ----------   | ------------ | -------    | -----     | ------     |
| **ESR-EA** | **ESRN-V-1** | **1.32**     | **40.616** | **31.65** | **0.8814** |
|            | **ESRN-V-2** | **1.31**     | **40.21**  | **31.69** | **0.8829** |
|            | **ESRN-V-3** | **1.31**     | **41.676** | **31.47** | **0.8803** |
|            | **ESRN-V-4** | **1.35**     | **40.17**  | **31.58** | **0.8814** |
| **SR_EA**  | **M2Mx2-A**  | **3.20**     | **196.27** | **32.20** | **0.8948** |
|            | **M2Mx2-B**  | **0.61**     | **35.03**  | **31.37** | **0.8796** |
|            | **M2Mx2-C**  | **0.24**     | **13.49**  | **30.92** | **0.8717** |
| SRCNN      | -            | -            | 52.7       | 29.50     | 0.8946     |
| CARN-M     | -            | -            | 91.2       | 31.23     | 0.9144     |
| FALSR-B    | -            | 0.32         | 74.70      | 31.28     | 0.9191     |

## Segmentation on VOC2012

| Method  | Model Name | Model Size/M | Flops/G | Params/K | mIOU   |
| ------- | ---------- | ------------ | ------- | ------- | ------ |
| **Adelaide_EA** | - | **10.6** | **0.5784** | **3822.14** | **0.7602** |
| MV2 + LW RefineNet | - | - | 0.92 |  4163 | 0.7313 |

## Click-Through Rate Prediction on Avazu

| Method  | Model Name | Model Size/M | Accuracy |
| ------- | ---------- | -------| ------ |
| **auto_group** | - | **111** | **0.790** |
| **auto_fis** | - | **500** | **0.788** |
| FM | - | 111 | 0.7793 |
| DeepFM | - | 111 | 0.7836 |
