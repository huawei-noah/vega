# CycleSR

## 1. 算法介绍

底层视觉任务中，由于很难在现实场景里获取到成对的数据，因此学术界大多采用人工合成的成对数据进行算法研究，然后通过合成数据获得到的算法模型往往在现实场景中表现并不好，主要原来来自于人工合成的降质数据无法比拟真实数据的分布。

Image2Image的风格迁移能够完成不同图片域的转换，CycleGAN就是其中一例得以成功应用的无监督风格迁移算法。

CycleSR是一种能够解决在非配对数据场景下图片超分任务的算法，该算法包含两部分网络--转换网络以及超分网络，和一种联合训练的策略，详细原理将在算法原理部分介绍。CycleSR巧妙地通过转换网络生成更贴合于真实场景的降质数据，从而与高清图片构成成对数据训练超分网络。

本算法适用于大多数无配对数据场景下的底层任务，使用者可以灵活的更改转换网络，替换超分网络为对应任务的模型，具体做法为只需要在base_model的基类上进行创建对应的模型类即可。

**Note: 本算法可以灵活的运用于很多非配对场景下的任务，其中转换网络部分可以理解为一种非配对下的data augmentation方式，后面的超分网络可以替换为其他任务网络。**

## 2. 算法原理

CycleSR包含转换网络和超分网络两部分，其网络结构图如下：

![CycleSR](../../images/cyclesr.png)

为了解决在没有配对数据场景下的超分问题，CycleSR的整体训练过程可分以下三个步骤：

1. 转换网络CycleGAN： 该网络能够完成两个域之间的风格转换，在这里是从高清域到真实低清域的迁移；
2. 超分网络： 1中生成的具有真实降质特性的图片与对应的高清图片构成成对的数据，从而实现监督性的训练超分网络；
3. 联合训练策略： 在整体训练过程中，联合训练转换网络和超分网络能够相互促进、提升各自的能力；在更新转换网络的生成器时，超分网络的损失会参与更新生成器参数，生成器的整体损失如下图所示。

![loss_trans](../../images/cyclesr_loss_trans.png)

### 2.1 网络配置

CycleSR的整体流程中只有一个fullytrain的过程, 其网络配置可参考`examples/data_augmentation/cyclesr/cyclesr.yml`, 按照训练过程总共可分为以下三点：

#### 转换网络

```yaml
clegan:
    input_nc: 3             # 输入channel数
    output_nc: 3            # 输出channel数
    ngf: 64                 # 生成器卷积filter的个数
    ndf: 64                 # 鉴别器卷积filter的个数
    n_layers_D: 3           # 鉴别器的卷积层个数
    norm: instance          # normalization的类型
    lambda_cycle: 10.0      # cycle loss 权重
    lambda_identity: 0.5    # identity loss权重
    buffer_size: 50         # Shuffle buffer大小
    up_mode: transpose      # 上采样类型
```

CycleGAN模型文件位于

```text
vega/networks/pytorch/cyclesrbodys/trans_model.py
```

#### 超分网络

```yaml
VDSR:
    name: VDSR              # SR 网络名称
    SR_nb: 20               # block的数量
    SR_nf: 64               # 卷积filter的个数
    SR_norm_type: batch     # 归一化方式 batch | instance | none
    upscale: 4              # 上采样因子
    input_nc: 3             # 模型输入通道数
    output_nc: 3            # 模型输出通道数
```

超分网络的模型文件位于:

```text
vega/networks/pytorch/cyclesrbodys/srmodels.py
```

#### 联合训练

```yaml
trainer:
    type: Trainer
    callbacks: CyclesrTrainerCallback
    lazy_built: True
    n_epoch: 100                # 学习率开始下降的epoch数
    n_epoch_decay: 100          # 学习率下降至0的epoch数
    val_ps_offset: 10           # 测试图片偏移量
    save_dir: /cache/logs/      # 保存路径
    lr_policy: linear           # 学习率类型

model:
    model_desc:
        modules: ["custom"]     # module 类型
        custom:
            type: CycleSRModel  # 模型名字
            SR_lam: 1000        # SR loss权重
            cycleSR_lam: 1000   # 用于更新cyclegan生成器的SR loss的权重
            grad_clip: 50       # 梯度clip阈值
            cyc_lr: !!float 2e-4 # cyclegan学习率
            SR_lr: !!float 1e-4  # SR网络学习率
```

## 3. 使用指导

可参考示例代码

算法的参数调整及运行配置通过配置文件的形式完成。该文件位于

```text
examples/data_augmentation/cyclesr/cyclesr.yml
```

## 4. 数据要求

对于HR需要将一张图片裁剪成像素为`480*480`的多张子图，对于LR需要将一张图片裁剪成像素为`120*120`的多张子图。

## 5. 算法输出

 1. 整个训练过程中产生的日志文件，包括控制台输出以及保存在Writer中的events
 2. 网络模型文件

最终在NTIRE17 track2上使用unpair的setting达到PSNR (RGB:25.34)
