**Vega ver1.0.0 released:**

- New algorithms: auto-lane, AutoFIS, AutoGroup, MFKD.
- Feature enhancement:
  - Trainer Callbacks: The trainer supports the callback mechanism and provides nine default callbacks.
  - Report mechanism: provides a unified data collection and processing mechanism for the AutoML algorithm.
  - Multi-Backend: TensorFlow is supported.
  - Evaluator Server: Provides independent evaluation services and model evaluation of Atlas 200DK and Bolt(coming soon).
- Community Contributors: qixiuai, hasanirtiza, sptj, cndylan, IlyaTrofimov.

**Introduction**

Vega is an AutoML algorithm tool chain developed by Noah's Ark Laboratory, the main features are as follows:

1. Full pipeline capailities: The AutoML capabilities cover key functions such as Hyperparameter Optimization, Data Augmentation, Network Architecture Search (NAS), Model Compression, and Fully Train. These functions are highly decoupled and can be configured as required, construct a complete pipeline.
2. Industry-leading AutoML algorithms: provides Noah's Ark Laboratory's self-developed **industry-leading algorithm** and  **Model Zoo** to download the State-of-the-art (SOTA) model.
3. High-concurrency neural network training capability: Provides high-performance trainers to accelerate model training and evaluation.
4. Multi-Backend: PyTorch, TensorFlow(trial), MindSpore(coming soon)

**Installation**

Before installation, you need to install some mandatory software packages. Please download the script <https://github.com/huawei-noah/vega/blob/master/deploy/install_dependencies.sh> and install them.

```bash
bash ./install_dependencies.sh
```

Install vega:

```bash
pip3 install noah-vega
```

**Cooperation and contribution**

Welcome to use Vega. If you have any questions, ask for help, fix bugs, contribute algorithms, or improve documents, submit the issue in the community. We will reply to and communicate with you in a timely manner.
Welcome to join our QQ chatroom (Chinese): **833345709**.
